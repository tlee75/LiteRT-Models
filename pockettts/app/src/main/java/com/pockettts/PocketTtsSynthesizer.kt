package com.pockettts

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Random
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pocket TTS (Kyutai, 100M) on LiteRT CompiledModel.
 *
 * Pocket TTS is a flow-matching LM over continuous 32-dim Mimi latents: per
 * 12.5 Hz frame a 6-layer/1024-wide causal transformer conditions a 6-block
 * AdaLN MLP flow head that turns one Gaussian draw into the next latent
 * (Lagrangian Self Distillation, 1 step — no iterative sampling loop); a 20M
 * tiny Mimi (x16 ConvTranspose upsample + 2-layer transformer + SEANet)
 * decodes latents to 24 kHz audio. The voice is a precomputed prompt KV cache
 * (`pt_voice_*.bin`, repacked from Kyutai's published per-voice states).
 *
 * Four graphs, all stateless with host-side state (the dia2/vibevoice
 * packed-KV pattern):
 *  * `pt_flowlm_step`  — one AR step; packed KV `[1,96,512,64]` in/out.
 *  * `pt_flow_head`    — cond + noise -> latent (LSD time embeds baked in).
 *  * `pt_mimi_dec_tx`  — 64-latent-frame block of the Mimi decoder
 *    transformer; blocks overlap 32 frames because the 2-layer sliding-window
 *    (250) attention has a stacked receptive field of 498 positions.
 *  * `pt_mimi_deconly` — SEANet decoder, one-shot 256-frame window (causal,
 *    so real frames are exact regardless of the zero tail).
 *
 * Text chunking, EOS handling and the noise schedule mirror the reference
 * pocket_tts Python package; the sentencepiece unigram tokenizer is ported in
 * [SpTokenizer]. Host-vs-reference parity of every graph and of the full
 * pipeline is checked in scripts/build_pockettts.py.
 */
class PocketTtsSynthesizer(context: Context) : Closeable {

    companion object {
        const val H = 1024               // flow-LM width
        const val HD = 64                // head dim
        const val NH = 16                // heads
        const val LAYERS = 6
        const val G = LAYERS * NH        // packed KV groups
        const val PMAX = 512             // KV capacity: voice + text + audio frames
        const val LDIM = 32              // Mimi latent dim
        const val THETA = 10000.0

        const val UPS = 16               // 12.5 Hz -> 200 Hz
        const val MIMI_D = 512
        const val F_BLK = 64             // dec_tx block payload frames
        const val F_HOP = 32             // dec_tx block hop
        const val S_BLK = F_BLK * UPS
        const val DEC_FRAMES = 256       // deconly window frames
        const val S_DEC = DEC_FRAMES * UPS
        const val SPF = 1920             // samples per 12.5 Hz frame
        const val SAMPLE_RATE = 24000

        // Generation defaults from the english config / pocket_tts defaults.
        const val TEMP = 0.3f
        const val EOS_THRESHOLD = -4.0f
        const val MAX_TOKENS_PER_CHUNK = 50
        const val TOKENS_PER_SECOND = 3.0
        const val GEN_SECONDS_PADDING = 2.0
        const val FRAME_RATE = 12.5
        const val MASK_NEG = -1e4f

        // step + flow head fused into one graph with one output tensor: on
        // Mali the per-frame cost is dispatch/sync-bound, and two invocations
        // plus four readbacks per frame cost more than the math itself.
        const val LM = "pt_flowlm_fused_fp16.tflite"
        const val DEC_TX = "pt_mimi_dec_tx_fp16.tflite"
        const val DECONLY = "pt_mimi_deconly_fp16.tflite"
        const val EMBED = "pt_embed_f16.bin"
        const val INPUT_LINEAR = "pt_input_linear_f32.bin"
        const val BOS = "pt_bos_input_f32.bin"
        const val NEUTRAL = "pt_neutral_latent_f32.bin"
        const val TOKENIZER = "pt_tokenizer.tsv"

        // CC-BY-4.0 (alba-mackenna, VCTK) and CC0 (voice-donations, voice-zero)
        // voices only; the CC-BY-NC ones (expresso, ears) are not bundled.
        val VOICES = listOf("alba", "marius", "javert", "charles", "mary", "eve")
    }

    private val modelDir =
        requireNotNull(context.getExternalFilesDir(null)) { "External storage unavailable" }

    private fun path(name: String): File {
        val f = File(modelDir, name)
        check(f.exists()) { "Missing $name — push files first: scripts/install_to_device.sh" }
        return f
    }

    // Debug overrides, read from the files dir (comma/newline-separated keys:
    // lm, dectx, dec). `force_cpu.txt` pins graphs to CPU; `force_fp32.txt`
    // keeps them on the GPU but at FP32 compute precision (the delegate's
    // default is fp16). Placement experiments only; absent in normal use.
    private fun overrideSet(file: String): Set<String> =
        File(modelDir, file).takeIf { it.exists() }
            ?.readText()?.split(',', '\n')?.map { it.trim() }?.filter { it.isNotEmpty() }
            ?.toSet() ?: emptySet()

    private val forceCpu = overrideSet("force_cpu.txt")
    private val forceFp32 = overrideSet("force_fp32.txt")

    // Keys that use force_gpu_opts.txt (normally just the LM).
    private val forceGpuOpts = overrideSet("force_gpu_opts_keys.txt")

    // `force_int8_lm.txt` present -> use *_int8.tflite graphs for LM + Mimi
    // instead of the fp16 files. The int8 graphs keep fp32 I/O so the host KV
    // contract is unchanged; used for CPU-only benchmarks.
    private val int8 = File(modelDir, "force_int8_lm.txt").exists()
    // `force_fp32_graph.txt` -> LM uses the full-fp32 graph (fp32 weights) for
    // speed comparisons vs the fp16-weight graph at FP32 compute (GPU32).
    private val fp32graph = File(modelDir, "force_fp32_graph.txt").exists()
    // `force_fp16_dec.txt` -> decoder (deconly) uses the fp16 graph even when
    // the LM/dectx are int8. Isolates the SEANet CPU-noise floor question:
    // "android XNNPACK computes native fp16 and collapses residual streams to
    // noise" (VibeVoice #64) — flip dec graph+placement while LM/dectx stay
    // int8-CPU and compare the silence.
    private val fp16dec = File(modelDir, "force_fp16_dec.txt").exists()
    // `force_fp32_dec.txt` -> decoder uses the full-fp32 graph. Test: does
    // fp32 SEANet on CPU avoid the residual-stream static entirely?
    private val fp32dec = File(modelDir, "force_fp32_dec.txt").exists()
    private val lmPath =
        when {
            int8 -> "pt_flowlm_fused_int8.tflite"
            fp32graph -> "pt_flowlm_fused.tflite"
            else -> LM
        }
    private val decTxPath = if (int8) "pt_mimi_dec_tx_int8.tflite" else DEC_TX
    private val deconlyPath =
        when {
            fp32dec -> "pt_mimi_deconly.tflite"
            int8 && !fp16dec -> "pt_mimi_deconly_int8.tflite"
            else -> DECONLY
        }

    // `force_dbg.txt` present -> per-step timing + latent/eos dumps. Independent
    // of which graph file is loaded (fp16/int8, GPU/CPU). Debug only.
    private val dbg = File(modelDir, "force_dbg.txt").exists()

    // `force_noise0.txt` present -> generation feeds ZERO noise (even with
    // TEMP>0). Discriminates whether the GPU fp16 NaN (seen at the first
    // generation frame, f=10) originates in the flow-HEAD's noise path
    // (clean when zeroed) or in the transformer body (still NaN when zeroed).
    private val noise0 = File(modelDir, "force_noise0.txt").exists()

    // `force_eps.txt` present -> in-memory override of the flow-LM LayerNorm
    // eps constant (float) before the graph is handed to the delegate. The
    // packed-LM keeps ALL 13 LayerNorms on ONE shared [1,1,1] fp32 constant
    // (`var + eps -> rsqrt`), and build_pockettts.py writes a sidecar
    // (`*.eps_offset`) with its flatbuffer byte offset. Default baked eps is
    // 1e-5 (subnormal in fp16); a subnormal-flushing GPU (Adreno) flushes it
    // to 0 -> rsqrt(0) -> NaN. fp16-safe values are >= 6.1e-5 (2^-14).
    // This patches the fp16 LM graph only; no rebuild / no duplicate graphs.
    private val forceEps: Float? =
        File(modelDir, "force_eps.txt").takeIf { it.exists() }?.readText()
            ?.trim()?.toFloatOrNull()

    private fun optTag(): String {
        val o = gpuOverrides()
        return "p=${o.precision?.name ?: "default"} b=${o.backend?.name ?: "auto"} " +
            "pl=${o.allowSrcQuantizedFcConvOps} tw=${o.preferTextureWeights} " +
            "cs=${o.constantTensorSharing} ic=${o.infiniteFloatCapping}"
    }

    // `force_gpu_opts.txt`: comma-separated overrides for LM GPU options,
    // for fp16-LM debugging on Adreno (none of these are the app default).
    // keys: precision=default|fp16|fp32, backend=automatic|opencl|webgpu|opengl,
    //       prec_loss=true|false, tex_weights=true|false, const_share=true|false,
    //       cap_ifinite=true|false
    private fun gpuOverrides(): CompiledModel.GpuOptions {
        val f = File(modelDir, "force_gpu_opts.txt").takeIf { it.exists() }
            ?.readText()?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() } ?: return CompiledModel.GpuOptions()
        var precision = CompiledModel.GpuOptions.Precision.DEFAULT
        var backend = CompiledModel.GpuOptions.Backend.AUTOMATIC
        var precLoss: Boolean? = null; var texW: Boolean? = null
        var constS: Boolean? = null; var capInf: Boolean? = null
        for (kv in f) {
            val parts = kv.split('=', limit = 2)
            if (parts.size < 2) continue
            val (k, v) = parts[0] to parts[1]
            when (k) {
                "precision" -> precision = when (v.lowercase()) {
                    "fp16" -> CompiledModel.GpuOptions.Precision.FP16
                    "fp32" -> CompiledModel.GpuOptions.Precision.FP32
                    else -> CompiledModel.GpuOptions.Precision.DEFAULT
                }
                "backend" -> backend = when (v.lowercase()) {
                    "opencl" -> CompiledModel.GpuOptions.Backend.OPENCL
                    "webgpu" -> CompiledModel.GpuOptions.Backend.WEBGPU
                    "opengl" -> CompiledModel.GpuOptions.Backend.OPENGL
                    else -> CompiledModel.GpuOptions.Backend.AUTOMATIC
                }
                "prec_loss" -> precLoss = v.toBoolean()
                "tex_weights" -> texW = v.toBoolean()
                "const_share" -> constS = v.toBoolean()
                "cap_ifinite" -> capInf = v.toBoolean()
            }
        }
        return CompiledModel.GpuOptions(
            constantTensorSharing = constS,
            infiniteFloatCapping = capInf,
            allowSrcQuantizedFcConvOps = null,
            precision = precision,
            bufferStorageType = null,
            preferTextureWeights = texW,
            serializationDir = null,
            modelCacheKey = null,
            serializeProgramCache = null,
            serializeExternalTensors = null,
            externalTensorsMode = null,
            externalTensorPattern = null,
            backend = backend,
            priority = null,
            numStepsOfCommandBufferPreparations = null,
        )
    }

    // CPU delegate options: the default is a single thread, which makes the
    // CPU path ~5x slower than it should be (the web bench's threaded wasm at
    // ~1.0 RTF). Use all available cores. Harmless on GPU builds (CPU only
    // used when force_cpu / fallback).
    private fun cpuOpts(): CompiledModel.Options {
        val opts = CompiledModel.Options(Accelerator.CPU)
        opts.cpuOptions =
            CompiledModel.CpuOptions(numThreads = Runtime.getRuntime().availableProcessors())
        return opts
    }

    // Temporary patched-graph copies (avoided unless force_eps.txt present).
    private val tempModels = ArrayList<File>()

    /** Patch the shared LayerNorm eps constant of `fp16Tflite` to [eps]. */
    private fun patchEps(fp16Tflite: File, eps: Float): File {
        val side = File(fp16Tflite.absolutePath + ".eps_offset")
        check(side.exists()) { "missing eps sidecar $side (rebuild with build_pockettts.py)" }
        val pos = side.readText().trim().split('\t').first().toInt()
        val data = RandomAccessFile(fp16Tflite, "r").use { f ->
            val b = ByteArray(f.length().toInt()); f.readFully(b); b
        }
        // little-endian float32 write in place
        val bits = java.lang.Float.floatToRawIntBits(eps)
        data[pos + 0] = (bits and 0xFF).toByte()
        data[pos + 1] = ((bits shr 8) and 0xFF).toByte()
        data[pos + 2] = ((bits shr 16) and 0xFF).toByte()
        data[pos + 3] = ((bits shr 24) and 0xFF).toByte()
        val tmp = File(modelDir, "pt_flowlm_fused_fp16_eps${eps}.tflite")
        RandomAccessFile(tmp, "rw").use { f -> f.write(data) }
        tempModels.add(tmp)
        return tmp
    }

    /** Compile on GPU; fall back to CPU (fp16 weights dequantize to fp32 there). */
    private fun load(name: String, key: String): Pair<CompiledModel, String> {
        val p = path(name).absolutePath
        if (key in forceCpu) {
            return CompiledModel.create(p, cpuOpts(), null) to "CPU*"
        }
        // eps override applies to the fp16 LM graph only (see force_eps.txt).
        val graphPath =
            if (key == "lm" && forceEps != null && name.endsWith("_fp16.tflite")) {
                patchEps(path(name), forceEps).absolutePath
            } else p
        return try {
            if (key in forceFp32) {
                val opts = CompiledModel.Options(Accelerator.GPU)
                opts.gpuOptions =
                    CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
                CompiledModel.create(graphPath, opts, null) to "GPU32"
            } else if (key in forceGpuOpts) {
                val opts = CompiledModel.Options(Accelerator.GPU)
                opts.gpuOptions = gpuOverrides()
                CompiledModel.create(graphPath, opts, null) to "GPU:${optTag()}"
            } else {
                CompiledModel.create(graphPath, CompiledModel.Options(Accelerator.GPU), null) to "GPU"
            }
        } catch (e: Throwable) {
            CompiledModel.create(p, cpuOpts(), null) to "CPU"
        }
    }

    // The Mimi decoder transformer runs on CPU BY DEFAULT: on Mali its GPU
    // output is audibly degraded (alba HNR 2.8 dB on CPU vs 0.9 dB on GPU,
    // recovered exactly to the fp32 eager level by this one move), and GPU
    // FP32 precision does NOT fix it — the same delegate behavior the Mimi
    // zoo module documents for its decoder transformer. It is 7 small calls
    // per utterance, so the speed cost is ~2% (1.03x -> 1.01x on a Pixel 8a).
    // `force_gpu.txt` with "dectx" re-enables GPU for experiments.
    private val forceGpu = overrideSet("force_gpu.txt")

    private val lmP = load(lmPath, "lm")
    private val dectxP =
        if ("dectx" in forceGpu) load(decTxPath, "dectx")
        else CompiledModel.create(
            path(decTxPath).absolutePath, cpuOpts(), null) to "CPU"
    private val deconlyP = load(deconlyPath, "dec")
    private val lm = lmP.first
    private val dectx = dectxP.first
    private val deconly = deconlyP.first

    /** e.g. "lm:GPU dectx:GPU dec:GPU" — shown in the UI status line. */
    val placements =
        "lm:${lmP.second} dectx:${dectxP.second} dec:${deconlyP.second}"

    private val lmIn = lm.createInputBuffers()
    private val lmOut = lm.createOutputBuffers()
    private val dectxIn = dectx.createInputBuffers()
    private val dectxOut = dectx.createOutputBuffers()
    private val deconlyIn = deconly.createInputBuffers()
    private val deconlyOut = deconly.createOutputBuffers()

    // ---- host assets ------------------------------------------------------
    private val embChannel = RandomAccessFile(path(EMBED), "r").channel
    private val embMap = embChannel
        .map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size()).order(ByteOrder.LITTLE_ENDIAN)
    private val inputLinear = readF32(path(INPUT_LINEAR))      // [1024, 32] row-major
    private val bosInput = readF32(path(BOS))                  // [1024]
    private val neutral = readF32(path(NEUTRAL))               // [32]
    val tokenizer = SpTokenizer(path(TOKENIZER))

    private val endTokens: Set<Int>
    private val fallbackTokens: Set<Int>

    init {
        endTokens = tokenizer.encode(".!...?").drop(1).toSet()
        fallbackTokens = tokenizer.encode(",;:").drop(1).toSet()
    }

    // ---- host state -------------------------------------------------------
    private val pk = FloatArray(G * PMAX * HD)
    private val pv = FloatArray(G * PMAX * HD)
    private val mask = FloatArray(NH * (PMAX + 1))
    private var pos = 0

    private var voiceName = ""
    private var voiceK = FloatArray(0)
    private var voiceV = FloatArray(0)
    private var voiceLen = 0

    private val cosArr = FloatArray(HD)
    private val sinArr = FloatArray(HD)
    private val invFreq = DoubleArray(HD / 2) { 1.0 / Math.pow(THETA, it / 32.0) }
    private val rnd = Random()

    data class Result(val audio: FloatArray, val frames: Int, val ms: Long)

    /** Load a repacked voice state: int32 T, then k and v as fp16 `[96][T][64]`. */
    fun loadVoice(name: String) {
        if (name == voiceName) return
        val bb = ByteBuffer.wrap(path("pt_voice_$name.bin").readBytes())
            .order(ByteOrder.LITTLE_ENDIAN)
        val t = bb.int
        check(t <= PMAX) { "voice state longer than KV capacity: $t > $PMAX" }
        val n = G * t * HD
        val k = FloatArray(n) { Half.toFloat(bb.short) }
        val v = FloatArray(n) { Half.toFloat(bb.short) }
        voiceK = k; voiceV = v; voiceLen = t; voiceName = name
    }

    private fun resetToVoice() {
        pk.fill(0f); pv.fill(0f)
        for (g in 0 until G) {
            System.arraycopy(voiceK, g * voiceLen * HD, pk, g * PMAX * HD, voiceLen * HD)
            System.arraycopy(voiceV, g * voiceLen * HD, pv, g * PMAX * HD, voiceLen * HD)
        }
        mask.fill(MASK_NEG)
        for (h in 0 until NH) {
            val base = h * (PMAX + 1)
            for (p in 0 until voiceLen) mask[base + p] = 0f
            mask[base + PMAX] = 0f                        // current token, concatenated at tail
        }
        pos = voiceLen
    }

    // ---- small host math --------------------------------------------------
    private fun embRow(id: Int): FloatArray {
        val out = FloatArray(H)
        var b = id * H * 2
        for (j in 0 until H) { out[j] = Half.toFloat(embMap.getShort(b)); b += 2 }
        return out
    }

    private fun projectLatent(lat: FloatArray): FloatArray {
        val out = FloatArray(H)
        for (o in 0 until H) {
            var acc = 0f
            val row = o * LDIM
            for (i in 0 until LDIM) acc += inputLinear[row + i] * lat[i]
            out[o] = acc
        }
        return out
    }

    private fun ropeFill(p: Int) {
        for (j in 0 until HD / 2) {
            val ang = p * invFreq[j]
            val c = cos(ang).toFloat(); val s = sin(ang).toFloat()
            cosArr[j] = c; cosArr[j + HD / 2] = c
            sinArr[j] = s; sinArr[j + HD / 2] = s
        }
    }

    private val zeroNoise = FloatArray(LDIM)

    /**
     * One fused frame: flow-LM step + flow head in a single invocation.
     * Output layout: eos(1) | latent(32) | new-k(96*64) | new-v(96*64).
     * Returns (latent, eosLogit) and appends this step's K/V at [pos].
     * Text prompting passes zero noise and ignores the latent.
     */
    // -- per-phase timing + latent dumps (only when force_dbg.txt present)
    private val tSetup = java.util.concurrent.atomic.AtomicLong(0)  // host prep
    private val tRun = java.util.concurrent.atomic.AtomicLong(0)    // lm.run
    private val tRead = java.util.concurrent.atomic.AtomicLong(0)   // readFloat
    private fun step(emb: FloatArray, noise: FloatArray): Pair<FloatArray, Float> {
        check(pos < PMAX) { "KV cache overflow at $pos" }
        val p0 = if (dbg) System.nanoTime() else 0L
        ropeFill(pos)
        val p1 = if (dbg) System.nanoTime() else 0L
        lmIn[0].writeFloat(emb)
        lmIn[1].writeFloat(cosArr)
        lmIn[2].writeFloat(sinArr)
        lmIn[3].writeFloat(mask)
        lmIn[4].writeFloat(pk)
        lmIn[5].writeFloat(pv)
        lmIn[6].writeFloat(noise)
        val p2 = if (dbg) System.nanoTime() else 0L
        lm.run(lmIn, lmOut)
        val p3 = if (dbg) System.nanoTime() else 0L
        val out = lmOut[0].readFloat()
        val p4 = if (dbg) System.nanoTime() else 0L
        if (dbg) {
            tSetup.addAndGet(p1 - p0 + p2 - p1)
            tRun.addAndGet(p3 - p2)
            tRead.addAndGet(p4 - p3)
        }
        val eos = out[0]
        val latent = out.copyOfRange(1, 1 + LDIM)
        if (dbg && (pos - voiceLen < 16)) {
            android.util.Log.i("PocketTTS",
                "dbg f=${pos - voiceLen} eos=$eos lat0=${latent[0]} lat1=${latent[1]} lat15=${latent[15]} lat31=${latent[31]} " +
                "latAbsMean=${latent.average()} latStd=${kotlin.math.sqrt(latent.map { it * it }.average())}")
            // also the raw output block (excluding the 6144*2 KV tail) to catch the
            // first NaN-bearing region precisely
            val nz = out.take(minOf(out.size, 1 + LDIM + 64)).filter { it.isNaN() }.size
            if (nz > 0) android.util.Log.i("PocketTTS", "dbg f=${pos - voiceLen} NaNcount(in front)=$nz firstNaNidx=" +
                out.take(1 + LDIM + 64).indexOfFirst { it.isNaN() })
        }
        val kvBase = 1 + LDIM
        for (g in 0 until G) {
            System.arraycopy(out, kvBase + g * HD, pk, g * PMAX * HD + pos * HD, HD)
            System.arraycopy(out, kvBase + G * HD + g * HD, pv, g * PMAX * HD + pos * HD, HD)
        }
        for (h in 0 until NH) mask[h * (PMAX + 1) + pos] = 0f
        pos++
        return latent to eos
    }

    /** Generate speech for `text` with the currently loaded voice. */
    fun synthesize(text: String, voice: String): Result {
        val t0 = System.nanoTime()
        loadVoice(voice)
        val audio = ArrayList<FloatArray>()
        var frames = 0
        val chunks = splitIntoBestSentences(text)
        for (chunk in chunks) {
            val (prepared, eosGuess) = prepareTextPrompt(chunk)
            val ids = tokenizer.encode(prepared)
            val latents = generateChunk(ids, framesAfterEos = eosGuess + 2)
            android.util.Log.i(
                "PocketTTS",
                "chunk: ${ids.size} tokens -> ${latents.size} frames",
            )
            frames += latents.size
            if (latents.isNotEmpty()) audio.add(decode(latents))
        }
        val total = audio.sumOf { it.size }
        val out = FloatArray(total)
        var o = 0
        for (a in audio) { System.arraycopy(a, 0, out, o, a.size); o += a.size }
        if (dbg) {
            android.util.Log.i("PocketTTS",
                "steps: setup=${tSetup.get()/1e6}ms run=${tRun.get()/1e6}ms read=${tRead.get()/1e6}ms tot=${(System.nanoTime()-t0)/1e6}ms")
        }
        return Result(out, frames, (System.nanoTime() - t0) / 1_000_000)
    }

    /** The reference autoregressive loop for one <=50-token chunk. */
    private fun generateChunk(ids: IntArray, framesAfterEos: Int): List<FloatArray> {
        resetToVoice()
        for (id in ids) step(embRow(id), zeroNoise)
        val estimate = ceil((ids.size / TOKENS_PER_SECOND + GEN_SECONDS_PADDING) * FRAME_RATE)
        val maxGen = minOf(estimate.toInt(), PMAX - pos - 1)
        val latents = ArrayList<FloatArray>(maxGen)
        var emb = bosInput
        var eosStep = -1
        for (g in 0 until maxGen) {
            // force_noise0.txt: feed ZERO noise even at generation. If the GPU
            // fp16 NaN disappears with zeroed noise but returns with real
            // noise, the fault is in the flow-HEAD's noise path; if it NaNs
            // either way, it is in the transformer body.
            val noise =
                if (noise0) zeroNoise
                else FloatArray(LDIM) {
                    (rnd.nextGaussian() * sqrt(TEMP.toDouble())).toFloat()
                }
            val (lat, eosLogit) = step(emb, noise)
            if (eosLogit > EOS_THRESHOLD && eosStep < 0) eosStep = g
            if (eosStep >= 0 && g >= eosStep + framesAfterEos) break
            latents.add(lat)
            emb = projectLatent(lat)
        }
        return latents
    }

    /** Mimi decode: overlapped dec_tx blocks -> one-shot SEANet window. */
    private fun decode(latents: List<FloatArray>): FloatArray {
        val t = minOf(latents.size, DEC_FRAMES)
        val feat = FloatArray(MIMI_D * S_DEC)
        val blk = FloatArray((1 + F_BLK) * LDIM)

        fun runBlock(prev: FloatArray, start: Int): FloatArray {
            System.arraycopy(prev, 0, blk, 0, LDIM)
            for (f in 0 until F_BLK) {
                val src = if (start + f < t) latents[start + f] else neutral
                System.arraycopy(src, 0, blk, (1 + f) * LDIM, LDIM)
            }
            dectxIn[0].writeFloat(blk)
            dectx.run(dectxIn, dectxOut)
            return dectxOut[0].readFloat()               // [512 * 1024]
        }

        var out = runBlock(neutral, 0)
        val n0 = minOf(F_BLK, t)
        for (c in 0 until MIMI_D)
            System.arraycopy(out, c * S_BLK, feat, c * S_DEC, n0 * UPS)
        var kept = F_BLK
        while (kept < t) {
            val start = kept - F_HOP
            out = runBlock(latents[start - 1], start)
            val n = minOf(F_BLK, t - start)
            val keepN = (n - F_HOP) * UPS
            for (c in 0 until MIMI_D)
                System.arraycopy(out, c * S_BLK + F_HOP * UPS, feat, c * S_DEC + kept * UPS, keepN)
            kept += n - F_HOP
        }

        deconlyIn[0].writeFloat(feat)
        deconly.run(deconlyIn, deconlyOut)
        val wav = deconlyOut[0].readFloat()
        return FloatArray(t * SPF) { wav[it].coerceIn(-1f, 1f) }
    }

    // ---- text preparation (ports of pocket_tts.models.tts_model) ----------

    /** prepare_text_prompt: normalize whitespace/case/punctuation; guess EOS tail. */
    internal fun prepareTextPrompt(raw: String): Pair<String, Int> {
        var text = raw.trim()
        require(text.isNotEmpty()) { "Text prompt cannot be empty" }
        text = text.replace('\n', ' ').replace('\r', ' ').replace("  ", " ")
        val words = text.trim().split(Regex("\\s+")).size
        val guess = if (words <= 4) 3 else 1
        if (!text[0].isUpperCase()) text = text[0].uppercaseChar() + text.substring(1)
        if (text.last().isLetterOrDigit()) text += "."
        return text to guess
    }

    /** split_into_best_sentences: sentence segments greedily packed <=50 tokens. */
    internal fun splitIntoBestSentences(raw: String): List<String> {
        val (prepared, _) = prepareTextPrompt(raw)
        val tokens = tokenizer.encode(prepared.trim()).toList()

        fun boundaries(list: List<Int>, marks: Set<Int>): List<Int> {
            val idx = ArrayList<Int>()
            idx.add(0)
            var prevWasBoundary = false
            for ((i, tok) in list.withIndex()) {
                if (tok in marks) prevWasBoundary = true
                else {
                    if (prevWasBoundary) idx.add(i)
                    prevWasBoundary = false
                }
            }
            idx.add(list.size)
            return idx
        }

        fun segments(list: List<Int>, idx: List<Int>): List<Pair<Int, String>> =
            (0 until idx.size - 1).map { i ->
                val part = list.subList(idx[i], idx[i + 1])
                part.size to tokenizer.decode(part)
            }

        val sentences = segments(tokens, boundaries(tokens, endTokens))
        val refined = ArrayList<Pair<Int, String>>()
        for ((n, textSeg) in sentences) {
            if (n <= MAX_TOKENS_PER_CHUNK) { refined.add(n to textSeg); continue }
            val sub = tokenizer.encode(textSeg.trim()).toList()
            val subSegs = segments(sub, boundaries(sub, fallbackTokens))
            if (subSegs.size > 1) refined.addAll(subSegs) else refined.add(n to textSeg)
        }

        val chunks = ArrayList<String>()
        var current = ""
        var count = 0
        for ((n, sentence) in refined) {
            when {
                current.isEmpty() -> { current = sentence; count = n }
                count + n > MAX_TOKENS_PER_CHUNK -> {
                    chunks.add(current.trim()); current = sentence; count = n
                }
                else -> { current += " $sentence"; count += n }
            }
        }
        if (current.isNotEmpty()) chunks.add(current.trim())
        return chunks
    }

    override fun close() {
        listOf(lmIn, lmOut, dectxIn, dectxOut, deconlyIn, deconlyOut)
            .forEach { l -> l.forEach { it.close() } }
        lm.close(); dectx.close(); deconly.close(); embChannel.close()
        tempModels.forEach { it.delete() }
        tempModels.clear()
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }
}
