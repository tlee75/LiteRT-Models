package com.pockettts

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Minimal Pocket TTS UI: pick a voice, type a sentence, tap Generate, listen.
 * Model load and generation run on a background thread; audio plays via
 * AudioTrack (float PCM) and the last output is saved to filesDir/output.wav.
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var synth: PocketTtsSynthesizer? = null
    private var rebuildSynthWithIntent = false

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var voices: Spinner
    private lateinit var profiles: Spinner
    private lateinit var button: Button
    private lateinit var waveform: WaveformView

    private fun gpuAvailable(): Boolean =
        try {
            val env = com.google.ai.edge.litert.Environment.create(this)
            env.getAvailableAccelerators().contains(com.google.ai.edge.litert.Accelerator.GPU)
        } catch (e: Throwable) { false }

    private fun selectedProfile(): Profile =
        Profile.entries.getOrElse(
            profiles.selectedItemPosition, { Profile.CPU_INT8W })

    /** Rebuild the synthesizer for the currently selected profile (spinner
     * change OR initial load). Runs on the background thread. */
    private fun rebuildSynth() {
        if (!::profiles.isInitialized) return
        runOnUiThread {
            button.isEnabled = false
            status.text = "Loading ${selectedProfile().label}…"
        }
        bg.execute {
            synth?.close()
            synth = try {
                PocketTtsSynthesizer(this, selectedProfile())
            } catch (e: Throwable) {
                android.util.Log.e("PocketTTS", "load failed", e)
                runOnUiThread { status.text = "Load failed: ${e.message}" }
                return@execute
            }
            android.util.Log.i("PocketTTS", "ready (${synth?.placements})")
            runOnUiThread {
                status.text = "Ready (${synth?.placements})."
                button.isEnabled = true
                if (rebuildSynthWithIntent) {
                    rebuildSynthWithIntent = false
                    runFromIntent(intent)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            setPadding(48, 48, 48, 48)
        }
        input = EditText(this).apply {
            hint = "Enter text to speak"
            setText("Hello! I am Pocket TTS, a tiny hundred million parameter model speaking to you from this phone.")
            minLines = 2
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                android.text.InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        voices = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                PocketTtsSynthesizer.VOICES,
            )
        }
        val gpuOk = gpuAvailable()
        profiles = Spinner(this).apply {
            // All three profiles shown; Hybrid greys out when no GPU is present.
            // isEnabled / getDropDownView drive the grey-out + non-selectable row.
            adapter = object : ArrayAdapter<Profile>(
                this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                Profile.entries,
            ) {
                override fun isEnabled(position: Int): Boolean {
                    val p = getItem(position)
                    return p != Profile.HYBRID || gpuOk
                }
                override fun getDropDownView(
                    position: Int, convertView: android.view.View?,
                    parent: android.view.ViewGroup?,
                ): android.view.View {
                    val v = super.getDropDownView(position, convertView, parent)
                    v.isEnabled = isEnabled(position)
                    v.alpha = if (isEnabled(position)) 1f else 0.4f
                    return v
                }
            }
            setSelection(0) // CPU (int8w/fp32) default
            onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: android.widget.AdapterView<*>?, v: android.view.View?,
                    pos: Int, id: Long,
                ) {
                    if (getItemAtPosition(pos) == Profile.HYBRID && !gpuOk) {
                        setSelection(0)
                        status.text = "Hybrid needs a GPU; using CPU (int8w/fp32)."
                    } else rebuildSynth()
                }
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            }
        }
        button = Button(this).apply { text = "Generate"; isEnabled = false }
        val modelsButton = Button(this).apply {
            text = "Models"
            setOnClickListener {
                startActivity(android.content.Intent(this@MainActivity, ModelsActivity::class.java))
            }
        }
        status = TextView(this).apply { text = "Loading model…"; textSize = 14f }
        waveform = WaveformView(this)
        val topMargins = intArrayOf(0, 24, 32, 24, 24, 24)
        for ((index, view) in listOf(input, voices, profiles, button, modelsButton, status).withIndex()) {
            val params = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            params.topMargin = topMargins[index]
            root.addView(view, params)
        }
        root.addView(waveform, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f).apply { topMargin = 24 })
        setContentView(root)

        rebuildSynthWithIntent = true
        rebuildSynth()

        button.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            val voice = voices.selectedItem as String
            button.isEnabled = false
            status.text = "Generating…"
            bg.execute {
                val s = synth ?: return@execute
                try {
                    val r = s.synthesize(text, voice)
                    saveWav(r.audio, voice)
                    val secs = r.audio.size.toFloat() / PocketTtsSynthesizer.SAMPLE_RATE
                    // Standard RTF = wall / audio (1.0 = real-time, lower = faster).
                    val rtf = r.ms / (secs * 1000f)
                    val line = "Spoke %.1fs (%d frames) in %d ms — RTF %.2f (%s)"
                        .format(secs, r.frames, r.ms, rtf, s.placements)
                    android.util.Log.i("PocketTTS", line)
                    // Play first; the waveform starts exactly when audio starts
                    // (inside play()) so the visual doesn't run ahead of the ear.
                    play(r.audio) {
                        runOnUiThread {
                            status.text = line
                            button.isEnabled = true
                            waveform.start(r.audio, PocketTtsSynthesizer.SAMPLE_RATE)
                        }
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "generation failed", e)
                    runOnUiThread { status.text = "Error: ${e.message}"; button.isEnabled = true }
                }
            }
        }
    }

    /** Headless driving: adb shell am start ... --es text "..." --es voice alba
     *  --es engine standard|studio|fast
     *  (singleTop, so a second am start generates again without reloading). */
    private fun runFromIntent(i: android.content.Intent?) {
        val t = i?.getStringExtra("text") ?: return
        input.setText(t)
        i.getStringExtra("voice")?.let { v ->
            val idx = PocketTtsSynthesizer.VOICES.indexOf(v)
            if (idx >= 0) voices.setSelection(idx)
        }
        i.getStringExtra("profile")?.let { e ->
            val idx = Profile.entries.indexOfFirst { it.name.equals(e, true) }
            if (idx >= 0) profiles.setSelection(idx)
        }
        if (button.isEnabled) button.performClick()
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        runFromIntent(intent)
    }

    /** Save the last output as a 24 kHz mono 16-bit WAV in filesDir (adb-pullable). */
    private fun saveWav(audio: FloatArray, voice: String) {
        val sr = PocketTtsSynthesizer.SAMPLE_RATE
        val data = audio.size * 2
        val bb = java.nio.ByteBuffer.allocate(44 + data).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + data); bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16); bb.putShort(1); bb.putShort(1)
        bb.putInt(sr); bb.putInt(sr * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(data)
        for (v in audio) bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        java.io.File(filesDir, "output.wav").writeBytes(bb.array())
        java.io.File(filesDir, "output_$voice.wav").writeBytes(bb.array())
    }

    /** Play `audio`, invoking [onStart] the moment playback begins so the
     *  waveform and the sound start together (not waveform-first). */
    private fun play(audio: FloatArray, onStart: () -> Unit) {
        if (audio.isEmpty()) return
        val track = AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
            AudioFormat.Builder()
                .setSampleRate(PocketTtsSynthesizer.SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
            audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        track.play()
        onStart()
        Thread.sleep((audio.size * 1000L / PocketTtsSynthesizer.SAMPLE_RATE) + 250)
        track.release()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        synth?.close()
    }
}
