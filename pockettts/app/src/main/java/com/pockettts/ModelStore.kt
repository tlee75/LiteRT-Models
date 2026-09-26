package com.pockettts

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Manages the on-device model/voice files the synthesizer needs.
 *
 * The app is not self-contained by design: the ~314 MB of graphs + voices are
 * fetched once (from the Pocket-TTS-LiteRT HF repo), stored in the app's
 * files dir, and reused until deleted. `PocketTtsSynthesizer.path()` checks
 * existence, so after a full download nothing is re-fetched.
 *
 * Every file named here is an adjacency list of what the synth calls: the
 * base files are required for any profile; the int8/fp32 variants are needed
 * by some profiles (see Engine). HF hosts the fp16 + fp32 + voices; the
 * int8 graphs we build are not on HF and are skipped with a note.
 */
class ModelStore(private val context: Context) {

    private val dir: File get() =
        requireNotNull(context.getExternalFilesDir(null)) { "External storage unavailable" }

    private val pool = Executors.newSingleThreadExecutor()

    data class Model(
        val name: String,          // flat file name in the synth's files dir
        val hfPath: String,        // path within the HF repo ("" if not hosted)
        val note: String,          // human explanation
        val onHf: Boolean = hfPath.isNotEmpty(),
    )

    val REQUIRED: List<Model> = listOf(
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
        // host assets (identical for every profile)
        Model("pt_embed_f16.bin", "pt_embed_f16.bin", "token embed table"),
        Model("pt_input_linear_f32.bin", "pt_input_linear_f32.bin", "latent projector"),
        Model("pt_bos_input_f32.bin", "pt_bos_input_f32.bin", "start-of-utterance vector"),
        Model("pt_neutral_latent_f32.bin", "pt_neutral_latent_f32.bin", "neutral latent"),
        Model("pt_tokenizer.tsv", "pt_tokenizer.tsv", "sentencepiece unigram table"),
    ) + PocketTtsSynthesizer.VOICES.map { v ->
        Model("pt_voice_$v.bin", "voices/pt_voice_$v.bin", "voice: $v")
    } + listOf(
        // int8 graphs — built locally, NOT hosted on HF. Needed for HYBRID.
        Model("pt_flowlm_fused_int8.tflite", "", "int8w LM (not on HF)"),
        Model("pt_mimi_dec_tx_int8.tflite", "", "int8 decoder tx (not on HF)"),
        Model("pt_mimi_deconly_int8.tflite", "", "int8 vocoder (not on HF)"),
    )

    fun localSize(name: String): Long = File(dir, name).takeIf { it.exists() }?.length() ?: -1

    fun isPresent(name: String): Boolean {
        val f = File(dir, name)
        return f.exists() && f.length() > 0
    }

    fun statusSummary(): String {
        val need = REQUIRED.count { it.hfPath.isNotEmpty() && !isPresent(it.name) }
        val have = REQUIRED.count { isPresent(it.name) }
        return if (need == 0) "All $have hosted models present."
        else "$need of $have+ hosted models missing — tap Download."
    }

    /** Download every hosted model the synth can use that isn't present yet.
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

    /** Delete every model file (managed + any orphan). */
    fun deleteAll() {
        REQUIRED.forEach { File(dir, it.name).delete() }
        // also the eps sidecar and any pt_* leftovers
        dir.listFiles()?.filter { it.name.startsWith("pt_") || it.name.endsWith(".eps_offset") }
            ?.forEach { it.delete() }
    }
}
