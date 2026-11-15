package pl.kinotts.studio.engine

import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig
import java.io.File
import java.util.UUID

/** Native in-process Supertonic 3 inference. No Python, Termux, or local HTTP server. */
class SupertonicEngine(private val outputDir: File) : AutoCloseable {
    /** Supertonic currently uses sherpa-onnx provider=cpu in this app. */
    @Volatile
    var executionProviderStatus: String = "CPU — Sherpa-ONNX provider=cpu (oczekuje na model)"
        private set

    private var tts: OfflineTts? = null
    private var loadedFrom: String? = null

    private val requiredFiles = listOf(
        "duration_predictor.int8.onnx",
        "text_encoder.int8.onnx",
        "vector_estimator.int8.onnx",
        "vocoder.int8.onnx",
        "tts.json",
        "unicode_indexer.bin",
        "voice.bin"
    )

    @Synchronized
    fun load(modelDir: File, numThreads: Int = 2) {
        require(numThreads in 1..8) { "Nieprawidłowa liczba wątków CPU" }
        val path = modelDir.canonicalPath
        if (tts != null && loadedFrom == path) return

        requiredFiles.forEach { name ->
            val file = File(modelDir, name)
            require(file.isFile && file.length() > 0L) {
                "Brak pliku modelu lub pusty plik: $name"
            }
        }

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

        val candidate = try {
            OfflineTts(config = cfg)
        } catch (e: NoClassDefFoundError) {
            throw IllegalStateException(
                "Brakuje klasy API Sherpa-ONNX w APK (np. OfflineTts): " +
                    (e.message ?: e.javaClass.simpleName) +
                    ". Sprawdź, czy zależność sherpa-onnx-jvm nie jest wykluczona.", e
            )
        } catch (e: UnsatisfiedLinkError) {
            throw IllegalStateException(
                "Nie można załadować JNI Sherpa-ONNX/ONNX Runtime (np. OrtGetApiBase): " +
                    (e.message ?: e.javaClass.simpleName) +
                    ". Sprawdź zgodność libonnxruntime.so z Sherpa-ONNX 2.0.1.", e
            )
        } catch (e: LinkageError) {
            throw IllegalStateException(
                "Błąd linkowania bibliotek lub klas Sherpa-ONNX: " +
                    (e.message ?: e.javaClass.simpleName), e
            )
        } catch (e: Exception) {
            throw IllegalStateException(
                "Nie udało się zainicjalizować Supertonic przez Sherpa-ONNX: " +
                    (e.message ?: e.javaClass.simpleName), e
            )
        }
        try {
            require(candidate.numSpeakers() >= 10) {
                "Model Supertonic ma mniej niż 10 stylów głosu"
            }
            require(candidate.sampleRate() > 0) { "Silnik zwrócił nieprawidłową częstotliwość próbkowania" }
        } catch (e: Exception) {
            try { candidate.free() } catch (_: Throwable) { }
            throw e
        } catch (e: LinkageError) {
            try { candidate.free() } catch (_: Throwable) { }
            throw IllegalStateException("Błąd natywny podczas sprawdzania modelu Supertonic.", e)
        }

        tts = candidate
        loadedFrom = path
        executionProviderStatus = "CPU — Sherpa-ONNX provider=cpu (potwierdzona konfiguracja silnika)"
    }

    @Synchronized
    fun generate(
        text: String,
        speakerId: Int,
        speed: Float,
        numSteps: Int,
        language: String
    ): File {
        require(text.isNotBlank()) { "Wpisz tekst do przeczytania" }
        require(language in SUPPORTED_LANGUAGES) { "Nieobsługiwany język: $language" }
        require(numSteps in 4..128) { "Liczba kroków musi mieścić się w zakresie 4–128" }

        val engine = tts ?: error("Model Supertonic nie został załadowany")
        val generated = engine.generateWithConfig(
            text,
            GenerationConfig(
                sid = speakerId.coerceIn(0, 9),
                speed = speed.coerceIn(0.6f, 1.6f),
                numSteps = numSteps,
                extra = mapOf("lang" to language)
            )
        )

        check(generated.samples.isNotEmpty()) { "Supertonic zwrócił pusty dźwięk" }
        check(generated.sampleRate > 0) { "Supertonic zwrócił nieprawidłowy sample rate" }
        check(outputDir.mkdirs() || outputDir.isDirectory) {
            "Nie można utworzyć katalogu na wygenerowane audio"
        }

        val file = File(outputDir, "kinotts_${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(8)}.wav")
        if (!generated.save(file.absolutePath) || !file.isFile || file.length() <= 44L) {
            file.delete()
            error("Nie udało się zapisać poprawnego pliku WAV")
        }
        return file
    }

    @Synchronized
    override fun close() {
        try {
            tts?.free()
        } catch (_: Exception) {
        } catch (_: LinkageError) {
        }
        tts = null
        loadedFrom = null
        executionProviderStatus = "CPU — Sherpa-ONNX provider=cpu (silnik odładowany)"
    }

    companion object {
        val SUPPORTED_LANGUAGES = setOf(
            "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
            "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "nl", "pl",
            "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi"
        )
    }
}
