package pl.kinotts.studio.xtts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import pl.kinotts.studio.engine.XtTsOnnxRuntime
import pl.kinotts.studio.engine.XtTsOnnxSessions
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.lang.reflect.Array as JArray
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.nio.charset.Charset
import java.util.LinkedHashMap
import java.util.LinkedHashSet
import java.util.Locale
import java.util.Random
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject

/** Cached zero-shot voice conditioning. Arrays are float32, row-major. */
data class XttsVoice internal constructor(
    internal val conditioning: FloatArray,       // [1, 32, 1024]
    internal val speakerEmbedding: FloatArray,  // [1, 512, 1]
    internal val sourcePath: String,
    internal val sourceModified: Long
)

data class XttsProgress(val fraction: Float, val stage: String)

data class XttsGptConfig(
    val layers: Int = 30,
    val embedding: Int = 1024,
    val heads: Int = 16,
    val headDim: Int = 64,
    val audioTokens: Int = 1026,
    val startAudio: Int = 1024,
    val stopAudio: Int = 1025,
    val condLength: Int = 32,
    val maxAudioTokens: Int = 605
)

/**
 * Fully local XTTS-v2 ONNX inference engine.
 * Pipeline: WAV -> 22.05kHz log-mel -> conditioning encoder; WAV -> 16kHz
 * speaker mel -> speaker encoder; BPE + GPT/KV loop -> HiFi-GAN -> 24kHz WAV.
 */
class XttsSynthesizer(
    private val modelDir: File,
    private val outputDir: File,
    private val numThreads: Int = 2,
    initialExecutionProvider: String = "cpu"
) : Closeable {
    private val cancelled = AtomicBoolean(false)
    private val loadLock = Any()
    @Volatile private var requestedExecutionProvider: String = initialExecutionProvider
    @Volatile var executionProviderStatus: String =
        if (initialExecutionProvider == "nnapi") "NNAPI — oczekuje na załadowanie modelu" else "CPU — gotowy"
        private set
    private val random = Random()
    private var env: OrtEnvironment? = null
    private var sessions: XtTsOnnxSessions? = null
    private var config = XttsGptConfig()
    private var startTextToken = -1
    private var stopTextToken = -1
    private var tokenizer: XttsBpe? = null
    private var melNorms = FloatArray(0)
    private var melEmbedding = FloatArray(0)
    private var melPosEmbedding = FloatArray(0)
    private var textEmbedding = FloatArray(0)
    private var textPosEmbedding = FloatArray(0)
    private var cachedVoice: XttsVoice? = null
    @Volatile private var closed = false

    /** Loads model tables and ONNX graphs once. Safe to call repeatedly. */
    fun open() {
        if (sessions != null) return
        synchronized(loadLock) {
            if (sessions != null) return
            check(!closed) { "Silnik XTTS został zamknięty." }
            require(modelDir.isDirectory) { "Nie znaleziono katalogu XTTS: ${modelDir.absolutePath}" }
            val metaFile = File(modelDir, "metadata.json")
            val meta = JSONObject(metaFile.readText(Charsets.UTF_8))
            config = XttsGptConfig(
                layers = meta.optInt("n_layer", 30),
                embedding = meta.optInt("embed_dim", 1024),
                heads = meta.optInt("num_heads", 16),
                headDim = meta.optInt("head_dim", 64),
                audioTokens = meta.optInt("num_audio_tokens", 1026),
                startAudio = meta.optInt("start_audio_token", 1024),
                stopAudio = meta.optInt("stop_audio_token", 1025),
                condLength = meta.optInt("perceiver_output_len", 32),
                maxAudioTokens = meta.optInt("max_gen_mel_tokens", 605)
            )
            require(config.layers in 1..48 && config.embedding == 1024 && config.heads == 16 && config.headDim == 64) {
                "Nieobsługiwana konfiguracja GPT w metadata.json."
            }
            require(config.condLength == 32 && config.audioTokens >= 1026) { "Nieprawidłowe wymiary XTTS." }
            val tokenizerFile = File(modelDir, "vocab.json")
            tokenizer = XttsBpe(tokenizerFile)
            startTextToken = tokenizer!!.tokenId("[START]")
            stopTextToken = tokenizer!!.tokenId("[STOP]")
            require(startTextToken >= 0 && stopTextToken >= 0) { "vocab.json nie zawiera [START] i [STOP]." }

            melNorms = Npy.readFloats(File(modelDir, "mel_stats.npy")).values
            require(melNorms.size == 80) { "mel_stats.npy musi mieć 80 wartości, ma ${melNorms.size}." }
            melEmbedding = readNpy(File(modelDir, "embeddings/mel_embedding.npy"), intArrayOf(1026, 1024))
            melPosEmbedding = readNpy(File(modelDir, "embeddings/mel_pos_embedding.npy"), intArrayOf(608, 1024))
            textEmbedding = readNpy(File(modelDir, "embeddings/text_embedding.npy"), intArrayOf(6681, 1024))
            textPosEmbedding = readNpy(File(modelDir, "embeddings/text_pos_embedding.npy"), intArrayOf(404, 1024))
            env = XtTsOnnxRuntime.environment()
            sessions = if (requestedExecutionProvider == "nnapi") {
                try {
                    XtTsOnnxRuntime.openSessions(modelDir, numThreads, useNnapi = true).also {
                        executionProviderStatus = "NNAPI EP zarejestrowany — Android/sterownik wybiera sprzęt; ORT nie rozróżnia tu GPU i NPU"
                    }
                } catch (error: Exception) {
                    XtTsOnnxRuntime.openSessions(modelDir, numThreads, useNnapi = false).also {
                        executionProviderStatus = "CPU fallback — NNAPI nie uruchomiło sesji: ${(error.message ?: error.javaClass.simpleName).take(120)}"
                    }
                } catch (error: LinkageError) {
                    XtTsOnnxRuntime.openSessions(modelDir, numThreads, useNnapi = false).also {
                        executionProviderStatus = "CPU fallback — brak biblioteki NNAPI: ${(error.message ?: error.javaClass.simpleName).take(120)}"
                    }
                }
            } else {
                XtTsOnnxRuntime.openSessions(modelDir, numThreads, useNnapi = false).also {
                    executionProviderStatus = "CPU — aktywny"
                }
            }
            validateGraphContract(sessions!!)
            outputDir.mkdirs()
        }
    }

    /** Changes the XTTS provider while idle; releases old sessions before reopening. */
    fun setExecutionProvider(provider: String) {
        require(provider == "cpu" || provider == "nnapi") { "Nieznany backend XTTS: $provider" }
        synchronized(loadLock) {
            check(!closed) { "Silnik XTTS został zamknięty." }
            if (requestedExecutionProvider == provider) return
            unloadLocked()
            requestedExecutionProvider = provider
            executionProviderStatus = if (provider == "nnapi")
                "NNAPI — oczekuje na załadowanie modelu" else "CPU — oczekuje na załadowanie modelu"
        }
    }

    /** Releases model tensors and sessions but keeps this synthesizer reusable. */
    fun unload() {
        synchronized(loadLock) {
            if (closed) return
            unloadLocked()
            executionProviderStatus = if (requestedExecutionProvider == "nnapi")
                "NNAPI — model odładowany" else "CPU — model odładowany"
        }
    }

    private fun unloadLocked() {
        cancelled.set(true)
        cachedVoice = null
        sessions?.close(); sessions = null
        tokenizer = null
        startTextToken = -1
        stopTextToken = -1
        melNorms = FloatArray(0)
        melEmbedding = FloatArray(0)
        melPosEmbedding = FloatArray(0)
        textEmbedding = FloatArray(0)
        textPosEmbedding = FloatArray(0)
        env = null
    }

    /** Cancels the current generation at the next inference boundary. */
    fun cancel() { cancelled.set(true) }

    /**
     * Creates/reuses speaker conditioning from a PCM16 reference WAV.
     * Reference audio is never uploaded or sent outside the device.
     */
    fun createVoice(referenceWav: File, onProgress: (XttsProgress) -> Unit = {}): XttsVoice {
        open()
        require(referenceWav.isFile && referenceWav.length() > 44L) { "Brak pliku WAV profilu głosu." }
        val stamp = referenceWav.lastModified()
        cachedVoice?.let {
            if (it.sourcePath == referenceWav.canonicalPath && it.sourceModified == stamp) return it
        }
        ensureActive()
        onProgress(XttsProgress(0.04f, "Odczyt próbki głosu"))
        val audio = Wav.readMonoPcm16(referenceWav)
        require(audio.samples.size >= audio.sampleRate / 2) { "Próbka głosu jest za krótka (minimum 0,5 s)." }
        val audio22 = Dsp.resample(audio.samples, audio.sampleRate, 22050).let { if (it.size > 22050 * 6) it.copyOf(22050 * 6) else it }
        require(audio22.size >= 2048) { "Próbka jest zbyt krótka do ekstrakcji cech." }
        onProgress(XttsProgress(0.10f, "Obliczanie cech conditioning"))
        val mel = Dsp.conditioningMel(audio22, melNorms)
        ensureActive()
        onProgress(XttsProgress(0.20f, "Encoder conditioning"))
        val cond = runFloatGraph(sessions!!.conditioningEncoder, "mel_spectrogram", mel, longArrayOf(1, 80, (mel.size / 80).toLong()))
        require(cond.size == 32 * 1024) { "Conditioning encoder zwrócił ${cond.size} wartości zamiast ${32 * 1024}." }

        ensureActive()
        onProgress(XttsProgress(0.26f, "Ekstrakcja embeddingu mówcy"))
        val audio16 = Dsp.resample(audio.samples, audio.sampleRate, 16000)
        val speakerMel = Dsp.speakerMel(audio16)
        val speaker = runFloatGraph(sessions!!.speakerEncoder, "mel_spec", speakerMel, longArrayOf(1, 64, (speakerMel.size / 64).toLong()))
        require(speaker.size == 512) { "Speaker encoder zwrócił ${speaker.size} wartości zamiast 512." }
        return XttsVoice(cond, speaker, referenceWav.canonicalPath, stamp).also { cachedVoice = it }
    }

    /** Full synthesis convenience API used by Studio. */
    fun synthesize(
        text: String,
        language: String,
        referenceWav: File,
        speed: Float = 1f,
        onProgress: (Float) -> Unit = {}
    ): File {
        cancelled.set(false)
        open()
        val voice = createVoice(referenceWav) { p -> onProgress(p.fraction) }
        return synthesize(text, language, voice, speed) { p -> onProgress(p) }
    }

    /** Synthesis with an already-computed voice profile. */
    fun synthesize(
        text: String,
        language: String,
        voice: XttsVoice,
        speed: Float = 1f,
        onProgress: (Float) -> Unit = {}
    ): File {
        cancelled.set(false)
        open()
        require(text.isNotBlank()) { "Wpisz tekst do syntezy." }
        require(speed in 0.5f..2f) { "Tempo musi być w zakresie 0,5–2,0." }
        require(voice.conditioning.size == 32 * 1024 && voice.speakerEmbedding.size == 512) { "Nieprawidłowy profil głosu." }
        val lang = language.substringBefore('-').lowercase(Locale.ROOT)
        require(lang in SUPPORTED_LANGUAGES) { "XTTS nie obsługuje języka: $lang" }
        val encoded = tokenizer!!.encode(text, lang)
        require(encoded.isNotEmpty()) { "Tokenizer nie zwrócił tokenów tekstu." }
        require(encoded.size + 2 <= 402) { "Tekst jest za długi dla XTTS. Podziel go na krótsze fragmenty." }
        ensureActive()
        onProgress(0.30f)
        val prefix = buildPrefix(voice.conditioning, encoded)
        val prefixLength = prefix.size / 1024
        var cache = Array(config.layers * 2) { FloatArray(0) }
        var cacheLength = 0
        var step = runGpt(prefix, prefixLength, cache, prefixLength)
        cache = step.cache
        cacheLength = prefixLength
        var logits = step.logits
        var hidden = step.hidden
        val generatedIds = ArrayList<Int>(config.maxAudioTokens)
        val latentVectors = ArrayList<FloatArray>(config.maxAudioTokens)
        val history = ArrayList<Int>(prefixLength + config.maxAudioTokens)
        repeat(prefixLength) { history.add(1) }
        var melPosition = 1
        val maximum = min(config.maxAudioTokens, 605)

        for (iteration in 0 until maximum) {
            ensureActive()
            val selected = sample(logits, history)
            if (selected == config.stopAudio) break
            if (selected !in 0 until 1024) error("GPT wygenerował nieprawidłowy kod audio: $selected")
            generatedIds.add(selected)
            latentVectors.add(hidden.copyOf())
            history.add(selected)
            onProgress(0.30f + 0.57f * ((iteration + 1f) / maximum))
            if (iteration == maximum - 1) break

            require(melPosition < 608) { "Przekroczono limit pozycji embeddingu audio XTTS." }
            val nextEmbedding = FloatArray(1024)
            val embOffset = selected * 1024
            val posOffset = melPosition * 1024
            for (j in 0 until 1024) nextEmbedding[j] = melEmbedding[embOffset + j] + melPosEmbedding[posOffset + j]
            melPosition++
            step = runGpt(nextEmbedding, 1, cache, cacheLength + 1)
            cache = step.cache
            cacheLength++
            logits = step.logits
            hidden = step.hidden
        }
        require(latentVectors.isNotEmpty()) { "XTTS nie wygenerował żadnych tokenów audio. Spróbuj inną próbkę lub tekst." }
        ensureActive()
        onProgress(0.90f)
        val latents = timeScale(latentVectors, speed)
        val waveform = runVocoder(latents, voice.speakerEmbedding)
        ensureActive()
        require(waveform.isNotEmpty()) { "HiFi-GAN zwrócił pusty dźwięk." }
        val outFile = File(outputDir, "xtts_${System.currentTimeMillis()}.wav")
        Wav.writePcm16(outFile, waveform, 24000)
        onProgress(1f)
        return outFile
    }

    private data class GptStep(val logits: FloatArray, val hidden: FloatArray, val cache: Array<FloatArray>)

    private fun buildPrefix(cond: FloatArray, tokens: List<Int>): FloatArray {
        val ids = ArrayList<Int>(tokens.size + 2)
        ids.add(startTextToken); ids.addAll(tokens); ids.add(stopTextToken)
        require(ids.size <= 404) { "Za dużo tokenów tekstowych." }
        val length = config.condLength + ids.size + 1
        val result = FloatArray(length * 1024)
        cond.copyInto(result, 0)
        var at = config.condLength * 1024
        for (i in ids.indices) {
            val id = ids[i]
            require(id in 0 until 6681) { "ID tokenu spoza tablicy embeddingów: $id" }
            val emb = id * 1024
            val pos = i * 1024
            for (j in 0 until 1024) result[at + j] = textEmbedding[emb + j] + textPosEmbedding[pos + j]
            at += 1024
        }
        val start = config.startAudio * 1024
        for (j in 0 until 1024) result[at + j] = melEmbedding[start + j] + melPosEmbedding[j]
        return result
    }

    private fun runGpt(input: FloatArray, inputLength: Int, cache: Array<FloatArray>, totalMaskLength: Int): GptStep {
        val session = sessions!!.gptDecoder
        val feed = LinkedHashMap<String, OnnxTensor>()
        val names = LinkedHashSet<String>()
        try {
            feed["inputs_embeds"] = tensor(input, longArrayOf(1, inputLength.toLong(), 1024))
            feed["attention_mask"] = tensor(FloatArray(totalMaskLength) { 1f }, longArrayOf(1, totalMaskLength.toLong()))
            for (layer in 0 until config.layers) {
                val pastKey = cache[layer * 2]
                val pastValue = cache[layer * 2 + 1]
                val pastLength = if (pastKey.isEmpty()) 0 else pastKey.size / (16 * 64)
                feed["past_key_$layer"] = tensor(pastKey, longArrayOf(1, 16, pastLength.toLong(), 64))
                feed["past_value_$layer"] = tensor(pastValue, longArrayOf(1, 16, pastLength.toLong(), 64))
                names.add("present_key_$layer"); names.add("present_value_$layer")
            }
            val outputs = LinkedHashSet<String>().apply {
                add("logits"); add("hidden_states"); addAll(names)
            }
            session.run(feed, outputs).use { result ->
                val logits = flatten(result.get("logits").get().value)
                val hiddenAll = flatten(result.get("hidden_states").get().value)
                require(logits.size >= config.audioTokens && hiddenAll.size >= 1024) { "Wyjście GPT ma nieprawidłowy kształt." }
                val newCache = Array(config.layers * 2) { FloatArray(0) }
                for (layer in 0 until config.layers) {
                    newCache[layer * 2] = flatten(result.get("present_key_$layer").get().value)
                    newCache[layer * 2 + 1] = flatten(result.get("present_value_$layer").get().value)
                }
                val lastLogits = logits.copyOfRange(logits.size - config.audioTokens, logits.size)
                val lastHidden = hiddenAll.copyOfRange(hiddenAll.size - 1024, hiddenAll.size)
                return GptStep(lastLogits, lastHidden, newCache)
            }
        } finally {
            feed.values.forEach { runCatching { it.close() } }
        }
    }

    private fun sample(source: FloatArray, history: List<Int>): Int {
        val scores = source.copyOf(config.audioTokens)
        val seen = HashSet<Int>(history)
        for (id in seen) if (id in scores.indices) scores[id] = if (scores[id] < 0f) scores[id] * 10f else scores[id] / 10f
        for (i in scores.indices) scores[i] /= 0.75f
        val sorted = scores.indices.sortedByDescending { scores[it] }
        val allowed = BooleanArray(scores.size)
        for (i in 0 until min(50, sorted.size)) allowed[sorted[i]] = true
        val maxLogit = scores.indices.filter { allowed[it] }.maxOf { scores[it] }
        var sum = 0.0
        val probs = DoubleArray(scores.size)
        for (i in scores.indices) if (allowed[i]) { probs[i] = exp((scores[i] - maxLogit).toDouble()); sum += probs[i] }
        if (sum <= 0.0 || !sum.isFinite()) return sorted.first()
        for (i in probs.indices) probs[i] /= sum
        val nucleus = sorted.filter { allowed[it] }
        var cumulative = 0.0
        val keep = HashSet<Int>()
        for (id in nucleus) {
            keep.add(id); cumulative += probs[id]
            if (cumulative >= 0.85) break
        }
        val keptSum = keep.sumOf { probs[it] }
        if (keptSum <= 0.0) return sorted.first()
        var draw = random.nextDouble() * keptSum
        for (id in sorted) if (id in keep) { draw -= probs[id]; if (draw <= 0.0) return id }
        return keep.first()
    }

    private fun runVocoder(latents: FloatArray, speaker: FloatArray): FloatArray {
        val session = sessions!!.hifiganVocoder
        val feed = linkedMapOf<String, OnnxTensor>()
        try {
            feed["latents"] = tensor(latents, longArrayOf(1, (latents.size / 1024).toLong(), 1024))
            feed["speaker_embedding"] = tensor(speaker, longArrayOf(1, 512, 1))
            session.run(feed).use { result ->
                return flatten(result.get(0).value)
            }
        } finally { feed.values.forEach { runCatching { it.close() } } }
    }

    private fun runFloatGraph(session: OrtSession, inputName: String, values: FloatArray, shape: LongArray): FloatArray {
        val input = tensor(values, shape)
        try { session.run(mapOf(inputName to input)).use { result -> return flatten(result.get(0).value) } }
        finally { input.close() }
    }

    private fun validateGraphContract(s: XtTsOnnxSessions) {
        val inputs = s.inputNames()
        require("mel_spectrogram" in inputs["conditioning_encoder"].orEmpty()) { "Conditioning ONNX nie ma wejścia mel_spectrogram." }
        require("mel_spec" in inputs["speaker_encoder"].orEmpty()) { "Speaker ONNX nie ma wejścia mel_spec." }
        val gpt = inputs["gpt_decoder"].orEmpty()
        require("inputs_embeds" in gpt && "attention_mask" in gpt && "past_key_0" in gpt && "past_value_0" in gpt) {
            "GPT ONNX nie ma oczekiwanych wejść inputs_embeds/attention_mask/KV-cache."
        }
        val gptOutputs = s.outputNames()["gpt_decoder"].orEmpty()
        require("logits" in gptOutputs && "hidden_states" in gptOutputs && "present_key_0" in gptOutputs && "present_value_0" in gptOutputs) {
            "GPT ONNX nie ma oczekiwanych wyjść logits/hidden_states/present KV-cache."
        }
        require("latents" in inputs["hifigan_vocoder"].orEmpty() && "speaker_embedding" in inputs["hifigan_vocoder"].orEmpty()) {
            "HiFi-GAN ONNX nie ma oczekiwanych wejść latents/speaker_embedding."
        }
    }

    private fun tensor(values: FloatArray, shape: LongArray): OnnxTensor {
        val expected = shape.fold(1L) { a, b -> Math.multiplyExact(a, b) }
        require(expected == values.size.toLong()) { "Tensor shape ${shape.contentToString()} nie pasuje do ${values.size} wartości." }
        val byteCount = max(1, values.size) * 4
        val buffer = ByteBuffer.allocateDirect(byteCount).order(ByteOrder.nativeOrder()).asFloatBuffer()
        if (values.isNotEmpty()) buffer.put(values)
        buffer.position(0); buffer.limit(values.size)
        return OnnxTensor.createTensor(env!!, buffer, shape)
    }

    private fun flatten(value: Any?): FloatArray {
        // First count, then fill one primitive array. Avoids ArrayList<Float> boxing
        // for the large GPT KV-cache tensors returned at every decoding step.
        var itemCount = 0
        fun countItems(v: Any?) {
            when (v) {
                null -> Unit
                is Number -> itemCount = Math.addExact(itemCount, 1)
                is FloatArray -> itemCount = Math.addExact(itemCount, v.size)
                is DoubleArray -> itemCount = Math.addExact(itemCount, v.size)
                else -> {
                    if (!v.javaClass.isArray) error("Nieobsługiwany typ wyjścia ONNX: ${v.javaClass.name}")
                    for (i in 0 until JArray.getLength(v)) countItems(JArray.get(v, i))
                }
            }
        }
        countItems(value)
        val out = FloatArray(itemCount)
        var offset = 0
        fun writeItems(v: Any?) {
            when (v) {
                null -> Unit
                is Number -> out[offset++] = v.toFloat()
                is FloatArray -> {
                    v.copyInto(out, destinationOffset = offset)
                    offset += v.size
                }
                is DoubleArray -> for (item in v) out[offset++] = item.toFloat()
                else -> for (i in 0 until JArray.getLength(v)) writeItems(JArray.get(v, i))
            }
        }
        writeItems(value)
        check(offset == itemCount) { "Niepełne spłaszczenie wyniku ONNX." }
        return out
    }

    private fun readNpy(file: File, expectedShape: IntArray): FloatArray {
        val array = Npy.readFloats(file)
        require(array.shape.contentEquals(expectedShape)) {
            "Nieprawidłowy kształt ${file.name}: ${array.shape.contentToString()}, oczekiwano ${expectedShape.contentToString()}."
        }
        return array.values
    }

    private fun timeScale(latents: List<FloatArray>, speed: Float): FloatArray {
        val count = latents.size
        val outCount = max(1, floor(count / speed.toDouble()).toInt())
        if (outCount == count) {
            val out = FloatArray(count * 1024)
            var offset = 0
            for (latent in latents) {
                require(latent.size == 1024) { "Nieprawidłowy wymiar wektora latentów." }
                latent.copyInto(out, destinationOffset = offset)
                offset += latent.size
            }
            return out
        }
        val out = FloatArray(outCount * 1024)
        for (t in 0 until outCount) {
            val source = if (outCount == 1) 0.0 else t.toDouble() * (count - 1) / (outCount - 1)
            val left = floor(source).toInt().coerceIn(0, count - 1)
            val right = min(count - 1, left + 1)
            val fraction = (source - left).toFloat()
            val a = latents[left]; val b = latents[right]
            val offset = t * 1024
            for (c in 0 until 1024) out[offset + c] = a[c] + (b[c] - a[c]) * fraction
        }
        return out
    }

    private fun ensureActive() { if (cancelled.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Synteza XTTS została anulowana.") }

    override fun close() {
        synchronized(loadLock) {
            if (closed) return
            unloadLocked()
            closed = true
            executionProviderStatus = "Zamknięty"
        }
    }

    companion object {
        val SUPPORTED_LANGUAGES = setOf("ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh")
    }
}

/** Small NPY v1/v2/v3 reader for float32/float16 model tables. */
private object Npy {
    data class ArrayData(val shape: IntArray, val values: FloatArray)
    fun readFloats(file: File): ArrayData {
        require(file.isFile && file.length() > 16) { "Brak lub pusty plik NPY: ${file.path}" }
        RandomAccessFile(file, "r").use { raf ->
            val magic = ByteArray(6); raf.readFully(magic)
            require(magic[0] == 0x93.toByte() && String(magic, 1, 5, Charsets.US_ASCII) == "NUMPY") { "Nieprawidłowy plik NPY: ${file.name}" }
            val major = raf.readUnsignedByte(); raf.readUnsignedByte()
            val headerLength = if (major == 1) readU16LE(raf) else readI32LE(raf)
            require(headerLength in 1..65536) { "Nieprawidłowy nagłówek NPY." }
            val headerBytes = ByteArray(headerLength); raf.readFully(headerBytes)
            val header = String(headerBytes, Charsets.UTF_8)
            val descr = Regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]").find(header)?.groupValues?.get(1)
                ?: error("Brak descr w ${file.name}")
            val fortran = Regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)").find(header)?.groupValues?.get(1) == "True"
            require(!fortran) { "Fortran-order NPY nie jest obsługiwany: ${file.name}" }
            val shapeText = Regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1)
                ?: error("Brak shape w ${file.name}")
            val shape = shapeText.split(',').mapNotNull { it.trim().takeIf(String::isNotEmpty)?.toIntOrNull() }.toIntArray()
            require(shape.isNotEmpty() && shape.all { it > 0 }) { "Nieprawidłowy shape w ${file.name}." }
            val count = shape.fold(1L) { a, b -> Math.multiplyExact(a, b.toLong()) }
            val bytesPer = when (descr) { "<f4", "=f4", "|f4", "<f", "=f" -> 4; "<f2", "=f2" -> 2; else -> error("Nieobsługiwany dtype NPY $descr (${file.name}).") }
            require(count <= Int.MAX_VALUE && raf.length() - raf.filePointer >= count * bytesPer) { "Dane NPY są ucięte: ${file.name}" }
            val offset = raf.filePointer
            val channel = raf.channel
            val mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, count * bytesPer).order(ByteOrder.LITTLE_ENDIAN)
            val result = FloatArray(count.toInt())
            if (bytesPer == 4) for (i in result.indices) result[i] = mapped.float
            else for (i in result.indices) result[i] = halfToFloat(mapped.short.toInt() and 0xffff)
            return ArrayData(shape, result)
        }
    }
    private fun readU16LE(raf: RandomAccessFile): Int { val a = raf.readUnsignedByte(); val b = raf.readUnsignedByte(); return a or (b shl 8) }
    private fun readI32LE(raf: RandomAccessFile): Int { val a=readU16LE(raf); val b=readU16LE(raf); return a or (b shl 16) }
    private fun halfToFloat(h: Int): Float {
        val sign = (h and 0x8000) shl 16
        var exp = (h ushr 10) and 0x1f
        var mant = h and 0x3ff
        val bits = when (exp) {
            0 -> if (mant == 0) sign else { exp = 1; while ((mant and 0x400) == 0) { mant = mant shl 1; exp-- }; mant = mant and 0x3ff; sign or ((exp + 112) shl 23) or (mant shl 13) }
            31 -> sign or 0x7f800000 or (mant shl 13)
            else -> sign or ((exp + 112) shl 23) or (mant shl 13)
        }
        return Float.fromBits(bits)
    }
}

/** PCM16 WAV reader/writer. Reference WAVs originate from AudioAnalyzer. */
private object Wav {
    data class Audio(val samples: FloatArray, val sampleRate: Int)
    fun readMonoPcm16(file: File): Audio {
        val bytes = file.readBytes()
        require(bytes.size >= 44) { "Plik WAV jest za krótki." }
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(ascii(bytes, 0, 4) == "RIFF" && ascii(bytes, 8, 4) == "WAVE") { "To nie jest plik WAV RIFF." }
        var pos = 12; var format = 0; var channels = 0; var rate = 0; var bits = 0; var dataAt = -1; var dataSize = 0
        while (pos + 8 <= bytes.size) {
            val id = ascii(bytes, pos, 4); val size = b.getInt(pos + 4).toLong() and 0xffffffffL
            val body = pos + 8
            if (size > Int.MAX_VALUE || body.toLong() + size > bytes.size) break
            when (id) {
                "fmt " -> if (size >= 16) { format = b.getShort(body).toInt() and 0xffff; channels = b.getShort(body + 2).toInt() and 0xffff; rate = b.getInt(body + 4); bits = b.getShort(body + 14).toInt() and 0xffff }
                "data" -> { dataAt = body; dataSize = size.toInt(); if (format != 0) break }
            }
            pos = body + size.toInt() + (size.toInt() and 1)
        }
        require(format == 1 && bits == 16 && channels in 1..8 && rate > 0 && dataAt >= 0) { "WAV referencyjny musi być PCM16." }
        val frames = dataSize / (2 * channels)
        val out = FloatArray(frames)
        var offset = dataAt
        for (i in 0 until frames) {
            var sum = 0f
            repeat(channels) { sum += b.getShort(offset).toFloat() / 32768f; offset += 2 }
            out[i] = (sum / channels).coerceIn(-1f, 1f)
        }
        return Audio(out, rate)
    }
    fun writePcm16(file: File, waveform: FloatArray, sampleRate: Int) {
        require(waveform.isNotEmpty() && waveform.size <= (Int.MAX_VALUE - 44) / 2) { "WAV wynikowy ma nieprawidłowy rozmiar." }
        file.parentFile?.mkdirs()
        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0); val dataBytes = waveform.size * 2
            val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            h.put("RIFF".toByteArray(Charsets.US_ASCII)); h.putInt(36 + dataBytes)
            h.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)); h.putInt(16); h.putShort(1); h.putShort(1)
            h.putInt(sampleRate); h.putInt(sampleRate * 2); h.putShort(2); h.putShort(16)
            h.put("data".toByteArray(Charsets.US_ASCII)); h.putInt(dataBytes); raf.write(h.array())
            val buf = ByteArray(8192); var at = 0
            while (at < waveform.size) {
                val n = min(buf.size / 2, waveform.size - at)
                for (i in 0 until n) {
                    val v = (waveform[at + i].takeIf { it.isFinite() } ?: 0f).coerceIn(-1f, 1f)
                    val q = (v * 32767f).roundToInt().coerceIn(-32768, 32767)
                    buf[i * 2] = q.toByte(); buf[i * 2 + 1] = (q shr 8).toByte()
                }
                raf.write(buf, 0, n * 2); at += n
            }
            raf.fd.sync()
        }
    }
    private fun ascii(b: ByteArray, at: Int, length: Int) = String(b, at, length, Charsets.US_ASCII)
}

/** FFT, windowing, Slaney conditioning mel, HTK speaker mel and sinc resampling. */
private object Dsp {
    fun resample(input: FloatArray, sourceRate: Int, targetRate: Int): FloatArray {
        if (sourceRate == targetRate) return input.copyOf()
        require(input.isNotEmpty() && sourceRate > 0 && targetRate > 0) { "Nieprawidłowe parametry resamplingu." }
        val ratio = targetRate.toDouble() / sourceRate
        val outLength = max(1, (input.size * ratio).roundToInt())
        val cutoff = min(1.0, ratio)
        val radius = ceil(20.0 / cutoff).toInt().coerceAtMost(160)
        val out = FloatArray(outLength)
        for (i in out.indices) {
            val center = i / ratio
            val first = max(0, floor(center).toInt() - radius + 1)
            val last = min(input.lastIndex, floor(center).toInt() + radius)
            var sum = 0.0; var weightSum = 0.0
            for (k in first..last) {
                val distance = center - k
                val x = distance * cutoff
                val sinc = if (abs(x) < 1e-9) 1.0 else sin(PI * x) / (PI * x)
                val windowArg = distance / radius
                if (abs(windowArg) >= 1.0) continue
                val window = 0.5 + 0.5 * cos(PI * windowArg)
                val weight = cutoff * sinc * window
                sum += input[k] * weight; weightSum += weight
            }
            out[i] = if (abs(weightSum) > 1e-12) (sum / weightSum).toFloat() else 0f
        }
        return out
    }

    fun conditioningMel(audio: FloatArray, norms: FloatArray): FloatArray {
        require(norms.size == 80)
        val mel = melSpectrogram(audio, 22050, 2048, 256, 1024, 80, 0f, 8000f, true, false)
        for (m in 0 until 80) for (t in 0 until mel.second) {
            val i = m * mel.second + t
            mel.first[i] = ln(max(1e-5f, mel.first[i])) / norms[m]
        }
        return mel.first
    }

    fun speakerMel(audio: FloatArray): FloatArray {
        require(audio.size > 512)
        val pre = FloatArray(audio.size)
        val first = audio[0]; val second = audio[1]
        val previous = 2f * first - second
        for (i in audio.indices) pre[i] = audio[i] - 0.97f * (if (i == 0) previous else audio[i - 1])
        return melSpectrogram(pre, 16000, 512, 160, 400, 64, 0f, 8000f, false, true).first
    }

    /** Returns row-major [mel, frame] data and frame count. */
    private fun melSpectrogram(
        audio: FloatArray, sr: Int, nfft: Int, hop: Int, winLength: Int, nMels: Int,
        fMin: Float, fMax: Float, slaney: Boolean, hamming: Boolean
    ): Pair<FloatArray, Int> {
        val paddedLength = audio.size + nfft
        val frames = 1 + (paddedLength - nfft) / hop
        val window = DoubleArray(winLength) { i ->
            val phase = 2.0 * PI * i / max(1, winLength)
            if (hamming) 0.54 - 0.46 * cos(phase) else 0.5 - 0.5 * cos(phase)
        }
        val winPad = DoubleArray(nfft)
        val leftPad = (nfft - winLength) / 2
        for (i in window.indices) winPad[leftPad + i] = window[i]
        val filters = buildMelFilters(nfft, sr, nMels, fMin, fMax, slaney)
        val out = FloatArray(nMels * frames)
        val real = DoubleArray(nfft); val imag = DoubleArray(nfft)
        val power = DoubleArray(nfft / 2 + 1)
        for (frame in 0 until frames) {
            val start = frame * hop - nfft / 2
            for (i in 0 until nfft) {
                val idx = reflectIndex(start + i, audio.size)
                real[i] = audio[idx].toDouble() * winPad[i]
                imag[i] = 0.0
            }
            fft(real, imag)
            for (f in power.indices) power[f] = real[f] * real[f] + imag[f] * imag[f]
            for (m in 0 until nMels) {
                var sum = 0.0
                for (f in power.indices) sum += power[f] * filters[f * nMels + m]
                out[m * frames + frame] = sum.toFloat()
            }
        }
        return out to frames
    }

    private fun reflectIndex(index: Int, length: Int): Int {
        require(length > 1) { "Próbka audio za krótka." }
        var i = index
        while (i < 0 || i >= length) {
            i = if (i < 0) -i else 2 * length - i - 2
        }
        return i
    }

    private fun hzToMel(hz: Double, slaney: Boolean): Double = if (!slaney) 2595.0 * ln(1.0 + hz / 700.0) / ln(10.0)
        else if (hz < 1000.0) 3.0 * hz / 200.0 else 15.0 + 27.0 * ln(hz / 1000.0) / ln(6.4)
    private fun melToHz(mel: Double, slaney: Boolean): Double = if (!slaney) 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
        else if (mel < 15.0) 200.0 * mel / 3.0 else 1000.0 * 6.4.pow((mel - 15.0) / 27.0)

    private fun buildMelFilters(nfft: Int, sr: Int, nMels: Int, fMin: Float, fMax: Float, slaney: Boolean): DoubleArray {
        val low = hzToMel(fMin.toDouble(), slaney); val high = hzToMel(fMax.toDouble(), slaney)
        val points = DoubleArray(nMels + 2) { i -> melToHz(low + (high - low) * i / (nMels + 1), slaney) }
        val bins = nfft / 2 + 1
        val result = DoubleArray(bins * nMels)
        for (f in 0 until bins) {
            val hz = f.toDouble() * sr / nfft
            for (m in 0 until nMels) {
                val left = points[m]; val centre = points[m + 1]; val right = points[m + 2]
                val rising = (hz - left) / max(1e-12, centre - left)
                val falling = (right - hz) / max(1e-12, right - centre)
                var value = max(0.0, min(rising, falling))
                if (slaney) value *= 2.0 / (right - left)
                result[f * nMels + m] = value
            }
        }
        return result
    }

    /** In-place radix-2 Cooley-Tukey FFT. */
    private fun fft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while ((j and bit) != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { val tr = real[i]; real[i] = real[j]; real[j] = tr; val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti }
        }
        var len = 2
        while (len <= n) {
            val angle = -2.0 * PI / len
            val wLenR = cos(angle); val wLenI = sin(angle)
            var base = 0
            while (base < n) {
                var wr = 1.0; var wi = 0.0
                for (k in 0 until len / 2) {
                    val even = base + k; val odd = even + len / 2
                    val vr = real[odd] * wr - imag[odd] * wi
                    val vi = real[odd] * wi + imag[odd] * wr
                    real[odd] = real[even] - vr; imag[odd] = imag[even] - vi
                    real[even] += vr; imag[even] += vi
                    val nextWr = wr * wLenR - wi * wLenI
                    wi = wr * wLenI + wi * wLenR; wr = nextWr
                }
                base += len
            }
            len = len shl 1
        }
    }
}

/** BPE tokenizer for the Hugging Face tokenizer.json stored as vocab.json. */
private class XttsBpe(file: File) {
    private val vocab = HashMap<String, Int>()
    private val ranks = HashMap<String, Int>()
    private var unknownId: Int = -1
    private val preTokenPattern = Regex("\\[[^\\]]+\\]|[\\p{L}\\p{M}\\p{N}_]+|[^\\p{L}\\p{M}\\p{N}_\\s]+")
    init {
        require(file.isFile) { "Brak vocab.json." }
        val root = JSONObject(file.readText(Charsets.UTF_8))
        val model = root.optJSONObject("model") ?: root
        val vocabJson = model.optJSONObject("vocab") ?: root.optJSONObject("vocab")
            ?: error("vocab.json nie zawiera model.vocab.")
        val keys = vocabJson.keys()
        while (keys.hasNext()) { val key = keys.next(); vocab[key] = vocabJson.getInt(key) }
        val added = root.optJSONArray("added_tokens") ?: JSONArray()
        for (i in 0 until added.length()) {
            val item = added.optJSONObject(i) ?: continue
            val content = item.optString("content")
            if (content.isNotEmpty()) vocab[content] = item.optInt("id", vocab[content] ?: -1)
        }
        unknownId = vocab[model.optString("unk_token", "[UNK]")] ?: vocab["[UNK]"] ?: -1
        val merges = model.optJSONArray("merges") ?: JSONArray()
        for (i in 0 until merges.length()) {
            val item = merges.opt(i)
            val pair = when (item) {
                is String -> item.trim().split(Regex("\\s+"), limit = 2)
                is JSONArray -> if (item.length() >= 2) listOf(item.getString(0), item.getString(1)) else emptyList()
                else -> emptyList()
            }
            if (pair.size == 2) ranks["${pair[0]}\u0001${pair[1]}"] = i
        }
        require(vocab.isNotEmpty() && ranks.isNotEmpty()) { "Nieprawidłowy BPE tokenizer w vocab.json." }
    }

    fun tokenId(token: String): Int = vocab[token] ?: -1

    fun encode(text: String, language: String): List<Int> {
        val cleaned = XttsPolish.clean(text, language)
        val tokenizerLanguage = if (language == "zh") "zh-cn" else language
        val marked = "[$tokenizerLanguage]" + cleaned.replace(" ", "[SPACE]")
        val result = ArrayList<Int>()
        for (match in preTokenPattern.findAll(marked)) {
            val piece = match.value
            val special = vocab[piece]
            if (piece.startsWith("[") && piece.endsWith("]") && special != null) {
                result.add(special)
            } else {
                bpe(piece).forEach { token ->
                    val id = vocab[token] ?: unknownId
                    require(id >= 0) { "Tokenizer nie zna fragmentu '$token'." }
                    result.add(id)
                }
            }
        }
        return result
    }

    private fun bpe(piece: String): List<String> {
        val cps = piece.codePoints().toArray()
        if (cps.isEmpty()) return emptyList()
        val parts = cps.map { String(Character.toChars(it)) }.toMutableList()
        while (parts.size > 1) {
            var bestRank = Int.MAX_VALUE; var bestIndex = -1
            for (i in 0 until parts.lastIndex) {
                val rank = ranks["${parts[i]}\u0001${parts[i + 1]}"] ?: Int.MAX_VALUE
                if (rank < bestRank) { bestRank = rank; bestIndex = i }
            }
            if (bestIndex < 0) break
            parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1]
            parts.removeAt(bestIndex + 1)
        }
        return parts
    }
}

/** Polish text cleanup adapted to the XTTS multilingual_cleaners pipeline. */
private object XttsPolish {
    private val ones = arrayOf("zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć")
    private val teens = arrayOf("dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście", "piętnaście", "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście")
    private val tens = arrayOf("", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt", "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt")
    private val hundreds = arrayOf("", "sto", "dwieście", "trzysta", "czterysta", "pięćset", "sześćset", "siedemset", "osiemset", "dziewięćset")

    fun clean(input: String, language: String): String {
        var text = input.trim().lowercase(Locale.ROOT).replace('’', '\'').replace('–', '-').replace("\"", "")
        if (language == "pl") {
            // Polish thousands are commonly separated by dots; strip only valid groups.
            text = Regex("\\b\\d{1,3}(?:\\.\\d{3})+\\b").replace(text) { it.value.replace(".", "") }
            text = Regex("(?:€\\s*([0-9]+(?:[.,][0-9]+)?)|([0-9]+(?:[.,][0-9]+)?)\\s*€)").replace(text) {
                currency(it.groupValues[1].ifEmpty { it.groupValues[2] }, "EUR")
            }
            text = Regex("(?:\\$\\s*([0-9]+(?:[.,][0-9]+)?)|([0-9]+(?:[.,][0-9]+)?)\\s*\\$)").replace(text) {
                currency(it.groupValues[1].ifEmpty { it.groupValues[2] }, "USD")
            }
            text = Regex("(?:£\\s*([0-9]+(?:[.,][0-9]+)?)|([0-9]+(?:[.,][0-9]+)?)\\s*£)").replace(text) {
                currency(it.groupValues[1].ifEmpty { it.groupValues[2] }, "GBP")
            }
            text = Regex("(?<![\\p{L}])([0-9]+)[,.]([0-9]+)(?![\\p{L}])").replace(text) { m ->
                integer(m.groupValues[1]) + " przecinek " + m.groupValues[2].map { ones[it.digitToInt()] }.joinToString(" ")
            }
            text = Regex("(?<![\\p{L}\\d])([0-9]+)(?![\\p{L}\\d])").replace(text) { integer(it.groupValues[1]) }
            val abbreviations = listOf(
                Regex("\\bp\\.") to "pani", Regex("\\bm\\.") to "pan",
                Regex("\\bdr\\.") to "doktor", Regex("\\bsw\\.") to "święty",
                Regex("\\bjr\\.") to "junior"
            )
            for ((pattern, replacement) in abbreviations) text = pattern.replace(text, replacement)
            text = text.replace("&", " i ").replace("@", " małpa ").replace("%", " procent ")
                .replace("#", " krzyżyk ").replace("$", " dolar ").replace("£", " funt ")
                .replace("°", " stopnie ").replace("€", " euro ")
        }
        return text.replace(Regex("\\s+"), " ").trim()
    }

    private fun currency(raw: String, kind: String): String {
        val parts = raw.replace(',', '.').split('.', limit = 2)
        val whole = parts[0].toLongOrNull() ?: return raw
        val unit = when (kind) {
            "EUR" -> "euro"
            "USD" -> plural(whole, "dolar", "dolary", "dolarów")
            else -> plural(whole, "funt", "funty", "funtów")
        }
        val base = integer(parts[0]) + " " + unit
        if (parts.size < 2) return base
        val centsText = parts[1].take(2).padEnd(2, '0')
        val cents = centsText.toIntOrNull() ?: 0
        if (cents == 0) return base
        return base + ", " + small(cents) + " " + plural(cents.toLong(), "cent", "centy", "centów")
    }

    private fun plural(n: Long, singular: String, few: String, many: String): String =
        when { n == 1L -> singular; n % 10 in 2L..4L && n % 100 !in 12L..14L -> few; else -> many }

    private fun integer(raw: String): String {
        val n = raw.toLongOrNull() ?: return raw.mapNotNull { it.digitToIntOrNull()?.let(ones::get) }.joinToString(" ")
        if (n < 0) return "minus " + integer((-n).toString())
        if (n < 1000) return small(n.toInt())
        if (n < 1_000_000) {
            val group = (n / 1000).toInt(); val tail = n % 1000
            val unit = plural(group.toLong(), "tysiąc", "tysiące", "tysięcy")
            return listOf(if (group == 1) "" else small(group), unit, if (tail == 0L) "" else small(tail.toInt())).filter { it.isNotEmpty() }.joinToString(" ")
        }
        if (n < 1_000_000_000) {
            val group = (n / 1_000_000).toInt(); val tail = n % 1_000_000
            val unit = plural(group.toLong(), "milion", "miliony", "milionów")
            return listOf(small(group), unit, if (tail == 0L) "" else integer(tail.toString())).filter { it.isNotEmpty() }.joinToString(" ")
        }
        if (n < 1_000_000_000_000) {
            val group = (n / 1_000_000_000).toInt(); val tail = n % 1_000_000_000
            val unit = plural(group.toLong(), "miliard", "miliardy", "miliardów")
            return listOf(small(group), unit, if (tail == 0L) "" else integer(tail.toString())).filter { it.isNotEmpty() }.joinToString(" ")
        }
        return raw
    }

    private fun small(n: Int): String {
        if (n < 10) return ones[n]
        if (n < 20) return teens[n - 10]
        if (n < 100) return listOf(tens[n / 10], if (n % 10 == 0) "" else ones[n % 10]).filter { it.isNotEmpty() }.joinToString(" ")
        return listOf(hundreds[n / 100], if (n % 100 == 0) "" else small(n % 100)).filter { it.isNotEmpty() }.joinToString(" ")
    }
}
