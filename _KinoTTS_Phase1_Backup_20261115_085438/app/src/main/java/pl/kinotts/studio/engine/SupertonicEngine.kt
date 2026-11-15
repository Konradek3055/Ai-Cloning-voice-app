package pl.kinotts.studio.engine

import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig
import java.io.File

/** Native in-process Supertonic 3 inference. No Python, Termux, or local HTTP server. */
class SupertonicEngine : AutoCloseable {
    private var tts: OfflineTts? = null
    private var loadedFrom: String? = null

    @Synchronized
    fun load(modelDir: File, numThreads: Int = 2) {
        val path = modelDir.canonicalPath
        if (tts != null && loadedFrom == path) return
        close()
        val cfg = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                supertonic = OfflineTtsSupertonicModelConfig(
                    durationPredictor = File(modelDir, "duration_predictor.int8.onnx").absolutePath,
                    textEncoder = File(modelDir, "text_encoder.int8.onnx").absolutePath,
                    vectorEstimator = File(modelDir, "vector_estimator.int8.onnx").absolutePath,
                    vocoder = File(modelDir, "vocoder.int8.onnx").absolutePath,
                    ttsJson = File(modelDir, "tts.json").absolutePath,
                    unicodeIndexer = File(modelDir, "unicode_indexer.bin").absolutePath,
                    voiceStyle = File(modelDir, "voice.bin").absolutePath
                ),
                numThreads = numThreads,
                debug = false,
                provider = "cpu"
            )
        )
        tts = OfflineTts(config = cfg)
        loadedFrom = path
    }

    @Synchronized
    fun generate(text: String, speakerId: Int, speed: Float, numSteps: Int, language: String): File {
        require(text.isNotBlank()) { "Wpisz tekst do przeczytania" }
        val engine = tts ?: error("Model Supertonic nie został załadowany")
        val generated = engine.generateWithConfig(
            text,
            GenerationConfig(
                sid = speakerId.coerceIn(0, 9),
                speed = speed.coerceIn(0.6f, 1.6f),
                numSteps = numSteps.coerceIn(4, 20),
                extra = mapOf("lang" to language)
            )
        )
        val dir = File(outputBaseDir(), "audio").apply { mkdirs() }
        val file = File(dir, "kinotts_${System.currentTimeMillis()}.wav")
        check(generated.save(file.absolutePath)) { "Nie udało się zapisać WAV" }
        return file
    }

    private fun outputBaseDir(): File {
        // The app's private storage root is set by the caller through the model path.
        return File(loadedFrom ?: error("Brak załadowanego modelu")).parentFile?.parentFile
            ?: error("Nie można ustalić katalogu aplikacji")
    }

    @Synchronized
    override fun close() {
        try { tts?.free() } catch (_: Exception) { }
        tts = null
        loadedFrom = null
    }
}
