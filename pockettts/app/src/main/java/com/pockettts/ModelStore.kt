package com.pockettts

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Manages the on-device files the synthesizer needs.
 *
 * The app is not self-contained by design: the ~380 MB of graphs + voices +
 * host assets are fetched once (from the Pocket-TTS-LiteRT HF repo), stored in
 * the app's files dir, and reused until deleted.
 * `PocketTtsSynthesizer.path()` checks existence, so after a full download
 * nothing is re-fetched. No HF token required at runtime: everything the app
 * downloads lives on public, ungated repos (the voice-clone encoder is baked
 * at BUILD time by `build_pockettts.py` from the ungated
 * `openensemble/pocket-tts` mirror of `kyutai/pocket-tts`).
 */
class ModelStore(private val context: Context) {

    private val dir: File get() =
        requireNotNull(context.getExternalFilesDir(null)) { "External storage unavailable" }

    private val pool = Executors.newSingleThreadExecutor()

    /** What a file is, used by the Files view to filter/label rows. */
    enum class Category(val label: String) {
        MODEL("Models"), VOICE("Voices"), ASSET("Host assets");

        companion object {
            fun of(name: String): Category = when {
                name.startsWith("pt_voice_") -> VOICE
                name.endsWith(".tflite") -> MODEL
                else -> ASSET
            }
        }
    }

    data class Model(
        val name: String,          // flat file name in the synth's files dir
        val hfPath: String,        // path within the HF repo ("" if not hosted)
        val note: String,          // human explanation
        val onHf: Boolean = hfPath.isNotEmpty(),
        val category: Category = Category.of(name),
    )

    val REQUIRED: List<Model> = listOf(
        // -- graphs (MODEL) ---------------------------------------------------
        Model("pt_flowlm_fused.tflite", "pt_flowlm_fused.tflite",
            "fp32 LM — CPU (FP32) profile"),
        Model("pt_flowlm_fused_fp16.tflite", "pt_flowlm_fused_fp16.tflite",
            "fp16 LM — debug/legacy"),
        Model("pt_mimi_dec_tx_fp16.tflite", "pt_mimi_dec_tx_fp16.tflite",
            "Mimi decoder transformer (any profile)"),
        Model("pt_mimi_deconly.tflite", "pt_mimi_deconly.tflite",
            "fp32 vocoder — CPU (int8w/fp32) + CPU (FP32)"),
        Model("pt_mimi_deconly_fp16.tflite", "pt_mimi_deconly_fp16.tflite",
            "fp16 vocoder — legacy"),
        // voice-clone reference encoder + prompt-BOS. Built from the UNGATED
        // openensemble/pocket-tts mirror at build time (build_pockettts.py
        // injects the live encoder — no gated repo / HF token needed);
        // distributed to users like the int8 graphs (adb push / our hosted
        // files once public), not bunded in the APK.
        Model("pt_mimi_encode.tflite", "", "voice-clone encoder (fp32) — build/push"),
        Model("pt_mimi_encode_fp16.tflite", "pt_mimi_encode_fp16.tflite",
            "voice-clone encoder (fp16)"),
        Model("pt_mimi_encode_int8.tflite", "pt_mimi_encode_int8.tflite",
            "voice-clone encoder (int8)"),
        // -- host assets (ASSET) ----------------------------------------------
        Model("pt_embed_f16.bin", "pt_embed_f16.bin", "token embed table"),
        Model("pt_input_linear_f32.bin", "pt_input_linear_f32.bin", "latent projector"),
        Model("pt_bos_input_f32.bin", "pt_bos_input_f32.bin", "start-of-utterance vector"),
        Model("pt_neutral_latent_f32.bin", "pt_neutral_latent_f32.bin", "neutral latent"),
        Model("pt_bos_before_voice_f32.bin", "", "clone prompt start-of-voice — build/push"),
        Model("pt_tokenizer.tsv", "pt_tokenizer.tsv", "sentencepiece unigram table"),
    ) + PocketTtsSynthesizer.VOICES.map { v ->
        Model("pt_voice_$v.bin", "voices/pt_voice_$v.bin", "preset voice: $v")
    } + listOf(
        // local-only builds (an HF token is required to build them from the
        // gated source; skip download with a note until we publish loudly).
        Model("pt_flowlm_fused_int8.tflite", "", "int8w LM (build locally)"),
        Model("pt_mimi_dec_tx_int8.tflite", "", "int8 decoder tx (build locally)"),
        Model("pt_mimi_deconly_int8.tflite", "", "int8 vocoder (build locally)"),
    )

    fun localSize(name: String): Long = File(dir, name).takeIf { it.exists() }?.length() ?: -1

    fun isPresent(name: String): Boolean {
        val f = File(dir, name)
        return f.exists() && f.length() > 0
    }

    fun statusSummary(): String {
        val need = REQUIRED.count { it.hfPath.isNotEmpty() && !isPresent(it.name) }
        val have = REQUIRED.count { isPresent(it.name) }
        return if (need == 0) "All $have hosted files present."
        else "$need of $have+ hosted files missing — tap Download."
    }

    /** Download every hosted file the synth can use that isn't present yet.
     *  Callbacks fire on the caller's chosen thread pool. */
    fun downloadMissing(onProgress: (done: Int, total: Int, name: String) -> Unit,
                        onDone: (downloaded: List<String>, failed: List<String>) -> Unit) {
        val todo = REQUIRED.filter { it.hfPath.isNotEmpty() && !isPresent(it.name) }
        val total = todo.size
        val downloaded = mutableListOf<String>()
        val failed = mutableListOf<String>()
        pool.execute {
            var done = 0
            for (m in todo) {
                onProgress(done, total, m.name)
                try {
                    fetch(m)
                    downloaded += m.name
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "download failed: ${m.name}", e)
                    failed += m.name
                }
                done++
                onProgress(done, total, m.name)
            }
            onDone(downloaded, failed)
        }
    }

    /** Fetch one hosted file into the flat files dir (voices/ renamed flat). */
    private fun fetch(m: Model) {
        val url = URL("https://huggingface.co/mlboydaisuke/Pocket-TTS-LiteRT/resolve/main/${m.hfPath}")
        val conn = url.openConnection() as HttpURLConnection
        conn.connectTimeout = 30000
        conn.readTimeout = 60000
        conn.instanceFollowRedirects = true
        try {
            check(conn.responseCode in 200..299) { "HTTP ${conn.responseCode}" }
            val out = File(dir, m.name)
            conn.inputStream.use { input ->
                out.outputStream().use { o -> input.copyTo(o, 1 shl 16) }
            }
            check(out.length() > 0) { "empty download" }
        } finally {
            conn.disconnect()
        }
    }

    /** Delete a stored file. */
    fun delete(name: String) { File(dir, name).delete() }

    /** Delete every file in the given categories (or all if none given). */
    fun deleteCategory(vararg cats: Category) {
        val names = REQUIRED.filter { it.category in cats }.map { it.name }.toSet()
        dir.listFiles()?.filter { it.name in names }
            ?.forEach { it.delete() }
    }

    /** Delete every managed file (any category) + the eps sidecar + orphans. */
    fun deleteAll() {
        val managed = REQUIRED.map { it.name }.toSet()
        dir.listFiles()?.filter {
            it.name in managed || it.name.startsWith("pt_") || it.name.endsWith(".eps_offset")
        }?.forEach { it.delete() }
    }
}
