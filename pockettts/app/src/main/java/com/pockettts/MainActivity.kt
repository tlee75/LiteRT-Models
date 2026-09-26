package com.pockettts

import android.app.Activity
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Audio Book UI: pick a voice, type a sentence, Generate, then Play (as many
 * times as you want). A "Clone a new voice…" entry in the voice picker opens
 * the system file picker; the clone runs fully on-device and becomes a
 * resident "custom" voice. Model load and generation run on a background
 * thread; the last output is saved to filesDir.
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var synth: PocketTtsSynthesizer? = null
    private var rebuildSynthWithIntent = false

    companion object {
        private const val REQ_CLONE = 11
        /** Terminal voice-picker entry that opens the clone file picker. */
        const val CLONE_ENTRY = "➕ Clone a new voice…"
    }

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var voices: Spinner
    private lateinit var profiles: Spinner
    private lateinit var generate: Button
    private lateinit var play: Button
    private lateinit var waveform: WaveformView
    private val clonePending = java.util.concurrent.atomic.AtomicBoolean(false)

    // Last generated audio (replayable via Play).
    private var lastAudio: FloatArray = FloatArray(0)

    private fun gpuAvailable(): Boolean =
        try {
            val env = com.google.ai.edge.litert.Environment.create(this)
            env.getAvailableAccelerators().contains(com.google.ai.edge.litert.Accelerator.GPU)
        } catch (e: Throwable) { false }

    private fun selectedProfile(): Profile =
        Profile.entries.getOrElse(
            profiles.selectedItemPosition, { Profile.CPU_INT8W })

    /** Voice names in the spinner: presets + (custom if cloned) + clone entry. */
    private fun voiceLabels(): List<String> =
        buildList {
            addAll(PocketTtsSynthesizer.VOICES)
            if (hasCustomVoice) add("custom")
            add(CLONE_ENTRY)
        }

    private var hasCustomVoice = false

    private fun refreshVoiceAdapter() {
        val pos = voices.selectedItemPosition
        voices.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, voiceLabels())
        // preserve selection when possible
        val keep = voices.adapter!!.getItem(pos)?.let { it } ?: CLONE_ENTRY
        val idx = voiceLabels().indexOf(keep)
        voices.setSelection(if (idx >= 0) idx else 0)
    }

    /** Rebuild the synthesizer for the currently selected profile (spinner
     * change OR initial load). Runs on the background thread. */
    private fun rebuildSynth() {
        if (!::profiles.isInitialized) return
        runOnUiThread {
            generate.isEnabled = false
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
                generate.isEnabled = true
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
                voiceLabels(),
            )
            // selecting the terminal clone entry opens the picker
            onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: android.widget.AdapterView<*>?, v: View?,
                    pos: Int, id: Long,
                ) {
                    if (voices.adapter?.getItem(pos) == CLONE_ENTRY && !clonePending.get()) {
                        openClonePicker()
                    }
                }
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            }
        }
        val gpuOk = gpuAvailable()
        profiles = Spinner(this).apply {
            // All three profiles shown; Hybrid greys out when no GPU is present.
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
                    parent: android.widget.AdapterView<*>?, v: View?,
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
        generate = Button(this).apply { text = "Generate"; isEnabled = false }
        play = Button(this).apply {
            text = "Play"; isEnabled = false
        }
        val filesButton = Button(this).apply {
            text = "Files"
            setOnClickListener {
                startActivity(Intent(this@MainActivity, FilesActivity::class.java))
            }
        }
        status = TextView(this).apply { text = "Loading model…"; textSize = 14f }
        waveform = WaveformView(this)
        // input, voices, profiles, generate, play, files, status, (waveform)
        val topMargins = intArrayOf(0, 24, 24, 32, 0, 24, 24)
        for ((index, view) in listOf(input, voices, profiles, generate, play, filesButton, status).withIndex()) {
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

        generate.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            val sel = voices.selectedItem as? String ?: return@setOnClickListener
            if (sel == CLONE_ENTRY) { openClonePicker(); return@setOnClickListener }
            generate.isEnabled = false
            status.text = "Generating…"
            bg.execute {
                val s = synth ?: return@execute
                try {
                    val r = s.synthesize(text, sel)
                    saveWav(r.audio, sel)
                    lastAudio = r.audio
                    val secs = r.audio.size.toFloat() / PocketTtsSynthesizer.SAMPLE_RATE
                    // Standard RTF = wall / audio (1.0 = real-time, lower = faster).
                    val rtf = r.ms / (secs * 1000f)
                    val line = "Spoke %.1fs (%d frames) in %d ms — RTF %.2f (%s)"
                        .format(secs, r.frames, r.ms, rtf, s.placements)
                    android.util.Log.i("PocketTTS", line)
                    runOnUiThread {
                        status.text = line
                        generate.isEnabled = true
                        play.isEnabled = true
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "generation failed", e)
                    runOnUiThread { status.text = "Error: ${e.message}"; generate.isEnabled = true }
                }
            }
        }

        // Separate Play button: enabled after generate (or clone preview), can
        // be tapped repeatedly to re-hear the last output.
        play.setOnClickListener {
            val audio = lastAudio
            if (audio.isNotEmpty()) playAudio(audio)
        }
    }

    private fun openClonePicker() {
        // AndroidX is not a dependency here; use the framework picker via
        // startActivityForResult (deprecated but functional).
        val pick = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "audio/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        try {
            startActivityForResult(pick, REQ_CLONE)
        } catch (e: Throwable) {
            status.text = "No file picker: ${e.message}"
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_CLONE) {
            if (resultCode == Activity.RESULT_OK && data?.data != null) {
                cloneFromUri(data.data!!)
            } else {
                // user cancelled: drop back to a preset selection
                voices.setSelection(0)
            }
        }
    }

    /** Read [uri], resample to mono, run the on-device clone, then select "custom". */
    private fun cloneFromUri(uri: Uri) {
        val bytes = try {
            java.io.FileInputStream(
                contentResolver.openFileDescriptor(uri, "r")!!.fileDescriptor)
                .use { it.readBytes() }
        } catch (e: Throwable) {
            status.text = "Clone failed: ${e.message}"
            voices.setSelection(0)
            return
        }
        cloneFromFile(java.io.File(filesDir, "clone_ref.wav").apply { writeBytes(bytes) })
    }

    /** Clone from a WAV on the app's filesDir, then select "custom". */
    private fun cloneFromFile(abs: java.io.File) {
        if (!clonePending.compareAndSet(false, true)) return
        generate.isEnabled = false
        status.text = "Cloning voice…"
        bg.execute {
            try {
                val wav = WavReader.read(abs.readBytes())
                val frames = synth?.cloneVoice(wav.samples, wav.sampleRate)
                    ?: throw IllegalStateException("synthesizer not ready")
                runOnUiThread {
                    android.util.Log.i("PocketTTS", "clone OK ($frames prompt frames)")
                    hasCustomVoice = true
                    refreshVoiceAdapter()
                    val idx = voiceLabels().indexOf("custom")
                    voices.setSelection(if (idx >= 0) idx else 0)
                    status.text = "Cloned voice ($frames prompt frames). Selected custom."
                    generate.isEnabled = true
                    // headless (--es ref): continue to generate once cloned
                    if (intent?.hasExtra("ref") == true &&
                        input.text.toString().isNotBlank()) {
                        generate.performClick()
                    }
                }
            } catch (e: Throwable) {
                android.util.Log.e("PocketTTS", "clone failed", e)
                runOnUiThread {
                    status.text = "Clone failed: ${e.message}"
                    generate.isEnabled = true
                    voices.setSelection(0)
                }
            } finally {
                clonePending.set(false)
            }
        }
    }

    /** Headless driving: adb shell am start ... --es text "..." --es voice alba
     *  --es profile cpu_int8w  [--es ref /path/to/ref.wav]
     *  `ref` clones from a WAV on the device files dir before generating.
     *  (singleTop, so a second am start generates again without reloading). */
    private fun runFromIntent(i: android.content.Intent?) {
        val t = i?.getStringExtra("text") ?: return
        input.setText(t)
        i.getStringExtra("voice")?.let { v ->
            if (v == "custom") hasCustomVoice = true
            refreshVoiceAdapter()
            val idx = voiceLabels().indexOf(v)
            if (idx >= 0) voices.setSelection(idx)
        }
        i.getStringExtra("profile")?.let { e ->
            val idx = Profile.entries.indexOfFirst { it.name.equals(e, true) }
            if (idx >= 0) profiles.setSelection(idx)
        }
        val ref = i.getStringExtra("ref")
        if (ref != null) {
            val f = java.io.File(filesDir, ref)
            if (f.exists()) cloneFromFile(f) else status.text = "ref not found: $ref"
        } else if (generate.isEnabled) {
            generate.performClick()
        }
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

    /** Play `audio`; the waveform starts exactly when audio starts. Replay-safe:
     *  each tap creates a fresh AudioTrack, so re-tapping just replays. */
    private fun playAudio(audio: FloatArray) {
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
        runOnUiThread { waveform.start(audio, PocketTtsSynthesizer.SAMPLE_RATE) }
        Thread { Thread.sleep((audio.size * 1000L / PocketTtsSynthesizer.SAMPLE_RATE) + 250); track.release() }
            .start()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        synth?.close()
    }
}