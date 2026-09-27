package com.pockettts

import android.app.Activity
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
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
    private val clonePending = java.util.concurrent.atomic.AtomicBoolean(false)

    // Last generated audio (replayable via Play).
    private var lastAudio: FloatArray = FloatArray(0)

    // Active playback; Play toggles to Stop while a track is live.
    private var track: AudioTrack? = null

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
        // input, voices, profiles, generate, play, files, status
        val topMargins = intArrayOf(0, 24, 24, 32, 0, 24, 24)
        for ((index, view) in listOf(input, voices, profiles, generate, play, filesButton, status).withIndex()) {
            val params = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            params.topMargin = topMargins[index]
            root.addView(view, params)
        }
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
                        autoplay()
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("PocketTTS", "generation failed", e)
                    runOnUiThread { status.text = "Error: ${e.message}"; generate.isEnabled = true }
                }
            }
        }

        // Play/Stop toggle: enabled after generate (or clone preview), tap
        // replays the last output, tap again stops playing.
        play.setOnClickListener {
            if (track != null) stopAudio()
            else {
                val audio = lastAudio
                if (audio.isNotEmpty()) playAudio(audio)
            }
        }
    }

    /** Stop the live AudioTrack if any and reset the button. */
    private fun stopAudio() {
        track?.let {
            it.pause()
            it.flush()
            it.release()
        }
        track = null
        play.text = "Play"
    }

    private fun autoplay() {
        val audio = lastAudio
        if (audio.isNotEmpty()) playAudio(audio)
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

    /** Read [uri], decode to mono float at its native rate, then clone. */
    private fun cloneFromUri(uri: Uri) {
        bg.execute {
            try {
                // MediaCodec decodes any container (mp3/m4a/ogg/wav); falls back
                // to the RIFF parser if the framework can't (e.g. odd PCM).
                val wav = decodeAudio(uri)
                    ?: WavReader.read(readUriBytes(uri))
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

    private fun readUriBytes(uri: Uri): ByteArray =
        java.io.FileInputStream(
            contentResolver.openFileDescriptor(uri, "r")!!.fileDescriptor)
            .use { it.readBytes() }

    /**
     * Generic audio decode via MediaExtractor+MediaCodec: returns mono float
     * PCM at the file's native sample rate (the synth linear-resamples to 24
     * kHz). A bounded one-shot reference (the encoder uses ~12.8 s max).
     * Uses a ParcelFileDescriptor data source (safe for both content:// and
     * file:// of app-private storage) with a hard iteration cap so it can
     * never hang. Returns null if the framework has no decoder for the track
     * (the caller falls back to [WavReader] for pure waved files).
     */
    private fun decodeAudio(uri: Uri): WavReader.Wav? {
        val pfd: ParcelFileDescriptor = try {
            contentResolver.openFileDescriptor(uri, "r") ?: return null
        } catch (e: Throwable) {
            android.util.Log.w("PocketTTS", "openFailed ${e.message}")
            return null
        }
        val extractor = android.media.MediaExtractor()
        var codec: android.media.MediaCodec? = null
        try {
            try {
                extractor.setDataSource(pfd.fileDescriptor, 0L,
                    pfd.statSize.takeIf { it >= 0 } ?: java.lang.Long.MAX_VALUE)
            } catch (e: Throwable) {
                android.util.Log.w("PocketTTS", "setDataSource ${e.message}")
                return null
            }
            var track = -1
            var fmt: android.media.MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val m = extractor.getTrackFormat(i)
                val mime = m.getString(android.media.MediaFormat.KEY_MIME)
                if (mime?.startsWith("audio/") == true) { track = i; fmt = m; break }
            }
            if (track < 0 || fmt == null) return null
            extractor.selectTrack(track)
            val sr = fmt.getInteger(android.media.MediaFormat.KEY_SAMPLE_RATE)
            val ch = fmt.getInteger(android.media.MediaFormat.KEY_CHANNEL_COUNT, 1)
            codec = try {
                android.media.MediaCodec.createDecoderByType(
                    fmt.getString(android.media.MediaFormat.KEY_MIME)!!)
            } catch (e: Throwable) {
                android.util.Log.w("PocketTTS", "no codec ${e.message}")
                return null
            }
            codec.configure(fmt, null, null, 0)
            codec.start()
            android.util.Log.i("PocketTTS", "decode: codec ${fmt.getString(android.media.MediaFormat.KEY_MIME)} start, sr=$sr ch=$ch")
            val info = android.media.MediaCodec.BufferInfo()
            val pcm = java.io.ByteArrayOutputStream(1 shl 20)
            var eof = false
            var guard = 0
            while (guard++ < 1_000_000) {
                if (!eof) {
                    var fed = 0
                    while (true) {
                        val ii = codec.dequeueInputBuffer(0)
                        if (ii < 0) break
                        val ib = codec.getInputBuffer(ii) ?: break
                        val n = extractor.readSampleData(ib, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(ii, 0, 0, 0,
                                android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            eof = true
                        } else {
                            val st = extractor.sampleTime
                            val sf = extractor.sampleFlags
                            // advance the extractor cursor to the NEXT sample
                            // (readSampleData alone does not move it)
                            extractor.advance()
                            codec.queueInputBuffer(ii, 0, n, st, sf)
                        }
                        fed++
                        if (eof) break
                        if (fed > 64) break   // bound: drain some before feeding more
                    }
                    if (fed > 64) android.util.Log.i("PocketTTS", "decode: fed burst $fed")
                }
                val oi = codec.dequeueOutputBuffer(info, if (eof) 20_000 else 100)
                if (oi >= 0) {
                    val ob = codec.getOutputBuffer(oi)
                    if (ob != null) {
                        ob.position(0); ob.limit(info.size)
                        ob.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        val b = ByteArray(info.size); ob.get(b); pcm.write(b)
                    }
                    val eos = info.flags and
                        android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    codec.releaseOutputBuffer(oi, false)
                    if (eos) {
                        android.util.Log.i("PocketTTS", "decode: ${pcm.size()}B pcm, sr=$sr ch=$ch")
                        return toMono(pcm.toByteArray(), ch, sr)
                    }
                } else if (oi == android.media.MediaCodec.INFO_TRY_AGAIN_LATER && eof) {
                    android.util.Log.i("PocketTTS", "decode: drained ${pcm.size()}B pcm, sr=$sr ch=$ch")
                    return toMono(pcm.toByteArray(), ch, sr)
                }
            }
            android.util.Log.w("PocketTTS", "decode guard hit")
            return toMono(pcm.toByteArray(), ch, sr)
        } catch (e: Throwable) {
            android.util.Log.w("PocketTTS", "decode failed ${e.message}")
            return null
        } finally {
            codec?.release()
            extractor.release()
            pfd.close()
        }
    }

    /** s16le interleaved [..c][..c] -> mono float [-1,1] (channel average). */
    private fun toMono(pcm: ByteArray, ch: Int, sr: Int): WavReader.Wav {
        val bb = java.nio.ByteBuffer.wrap(pcm).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val shorts = pcm.size / 2
        val frames = shorts / ch
        val f32 = FloatArray(frames) { i ->
            var acc = 0.0
            for (c in 0 until ch) acc += bb.getShort(2 * (i * ch + c)).toDouble() / 32768.0
            (acc / ch).toFloat()
        }
        return WavReader.Wav(f32, sr)
    }

    /** Clone from a WAV/audio on the app's filesDir, then select "custom". */
    private fun cloneFromFile(abs: java.io.File) {
        if (!clonePending.compareAndSet(false, true)) return
        generate.isEnabled = false
        status.text = "Cloning voice…"
        bg.execute {
            try {
                // file:// URI decode (MediaCodec for any container) + RIFF fallback
                val wav = decodeAudio(Uri.fromFile(abs))
                    ?: WavReader.read(abs.readBytes())
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
     *  the Play button toggles to Stop while playing; tapping again stops. */
    private fun playAudio(audio: FloatArray) {
        if (audio.isEmpty()) return
        stopAudio()
        val t = AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
            AudioFormat.Builder()
                .setSampleRate(PocketTtsSynthesizer.SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
            audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        t.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        t.play()
        track = t
        play.text = "Stop"
        Thread {
            Thread.sleep((audio.size * 1000L / PocketTtsSynthesizer.SAMPLE_RATE) + 250)
            // reset to "Play" only if this is still the live track
            runOnUiThread { if (track === t) { track = null; play.text = "Play" } }
            t.release()
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        synth?.close()
    }
}