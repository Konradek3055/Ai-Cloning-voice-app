package pl.kinotts.studio.engine

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File

/** Result of loading the Android ONNX Runtime native libraries. */
data class OrtRuntimeCheck(
    val ready: Boolean,
    val version: String?,
    val message: String
)

/**
 * ONNX Runtime bootstrap and XTTS graph-session owner.
 *
 * Maven Central Android AAR is 1.28.0; Sherpa-ONNX 2.0.1 release notes identify native ORT 1.28.2.
 * Use the matching Java/JNI version so both engines share one compatible
 * native ONNX Runtime library.
 *
 * This class loads XTTS graphs. It does not claim to synthesize speech by
 * itself: the BPE, feature extraction, GPT KV-cache loop and vocoder pipeline
 * must call these sessions in the next implementation layer.
 */
object XtTsOnnxRuntime {
    const val EXPECTED_VERSION = "1.28.x"

    @Volatile
    private var environment: OrtEnvironment? = null

    /** Loads the Android runtime, disables telemetry and verifies its version. */
    @Synchronized
    fun diagnose(): OrtRuntimeCheck {
        return try {
            val ort = environment ?: OrtEnvironment
                .getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
                .also {
                    it.setTelemetry(false)
                    environment = it
                }

            val actualVersion = ort.version
            if (!actualVersion.startsWith("1.28.")) {
                OrtRuntimeCheck(
                    ready = false,
                    version = actualVersion,
                    message = "Niezgodna biblioteka ONNX Runtime: $actualVersion. Wymagana: $EXPECTED_VERSION."
                )
            } else {
                OrtRuntimeCheck(
                    ready = true,
                    version = actualVersion,
                    message = "ONNX Runtime $actualVersion gotowy (CPU, telemetry wyłączona)."
                )
            }
        } catch (error: Exception) {
            OrtRuntimeCheck(
                ready = false,
                version = null,
                message = "Nie udało się uruchomić ONNX Runtime: ${error.message ?: error.javaClass.simpleName}"
            )
        } catch (error: LinkageError) {
            OrtRuntimeCheck(
                ready = false,
                version = null,
                message = "Błąd bibliotek natywnych ONNX Runtime: ${error.message ?: error.javaClass.simpleName}"
            )
        }
    }

    /** Public environment accessor for the XTTS inference layer. */
    fun environment(): OrtEnvironment = requireEnvironment()

    private fun requireEnvironment(): OrtEnvironment {
        val result = diagnose()
        check(result.ready) { result.message }
        return checkNotNull(environment)
    }

    /**
     * Opens the four XTTS-v2 ONNX graphs from an installed model pack.
     * Uses one CPU thread pool per graph, with a conservative 1..4 thread cap.
     */
    fun openSessions(modelDir: File, numThreads: Int = 2, useNnapi: Boolean = false): XtTsOnnxSessions {
        require(numThreads in 1..8) { "Liczba wątków XTTS musi mieścić się w zakresie 1–8." }
        require(modelDir.isDirectory) { "Brak katalogu modeli XTTS: ${modelDir.absolutePath}" }

        val conditioningFile = File(modelDir, "conditioning_encoder.onnx")
        val speakerFile = File(modelDir, "speaker_encoder.onnx")
        val vocoderFile = File(modelDir, "hifigan_vocoder.onnx")
        val gptInt8 = File(modelDir, "gpt_model_int8.onnx")
        val gptFile = if (gptInt8.isFile) gptInt8 else File(modelDir, "gpt_model.onnx")

        val requiredAssets = listOf(
            conditioningFile,
            speakerFile,
            vocoderFile,
            gptFile,
            File(modelDir, "metadata.json"),
            File(modelDir, "vocab.json"),
            File(modelDir, "mel_stats.npy"),
            File(modelDir, "embeddings/mel_embedding.npy"),
            File(modelDir, "embeddings/mel_pos_embedding.npy"),
            File(modelDir, "embeddings/text_embedding.npy"),
            File(modelDir, "embeddings/text_pos_embedding.npy")
        )
        val missing = requiredAssets.filterNot { it.isFile && it.length() > 0L }
        require(missing.isEmpty()) {
            "Niekompletny pakiet XTTS. Brak: " +
                missing.joinToString { it.relativeTo(modelDir).path }
        }

        val ort = requireEnvironment()
        val opened = mutableListOf<Pair<OrtSession, OrtSession.SessionOptions>>()

        fun loadSession(file: File): OrtSession {
            val options = OrtSession.SessionOptions()
            try {
                options.setIntraOpNumThreads(numThreads)
                options.setInterOpNumThreads(1)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                options.setCPUArenaAllocator(true)
                options.setMemoryPatternOptimization(true)
                if (useNnapi) options.addNnapi()
                val session = ort.createSession(file.absolutePath, options)
                opened.add(session to options)
                return session
            } catch (error: Throwable) {
                runCatching { options.close() }
                throw error
            }
        }

        try {
            val conditioning = loadSession(conditioningFile)
            val speaker = loadSession(speakerFile)
            val gpt = loadSession(gptFile)
            val vocoder = loadSession(vocoderFile)
            return XtTsOnnxSessions(
                conditioningEncoder = conditioning,
                speakerEncoder = speaker,
                gptDecoder = gpt,
                hifiganVocoder = vocoder,
                managed = opened.toList()
            )
        } catch (error: Throwable) {
            opened.asReversed().forEach { (session, options) ->
                runCatching { session.close() }
                runCatching { options.close() }
            }
            throw error
        }
    }
}

/** Keeps SessionOptions alive until the corresponding OrtSession is closed. */
class XtTsOnnxSessions internal constructor(
    val conditioningEncoder: OrtSession,
    val speakerEncoder: OrtSession,
    val gptDecoder: OrtSession,
    val hifiganVocoder: OrtSession,
    private val managed: List<Pair<OrtSession, OrtSession.SessionOptions>>
) : Closeable {
    @Volatile
    private var closed = false

    fun ensureOpen() {
        check(!closed) { "Sesje XTTS zostały już zamknięte." }
    }

    fun inputNames(): Map<String, Set<String>> {
        ensureOpen()
        return mapOf(
            "conditioning_encoder" to conditioningEncoder.inputInfo.keys,
            "speaker_encoder" to speakerEncoder.inputInfo.keys,
            "gpt_decoder" to gptDecoder.inputInfo.keys,
            "hifigan_vocoder" to hifiganVocoder.inputInfo.keys
        )
    }

    fun outputNames(): Map<String, Set<String>> {
        ensureOpen()
        return mapOf(
            "conditioning_encoder" to conditioningEncoder.outputInfo.keys,
            "speaker_encoder" to speakerEncoder.outputInfo.keys,
            "gpt_decoder" to gptDecoder.outputInfo.keys,
            "hifigan_vocoder" to hifiganVocoder.outputInfo.keys
        )
    }

    override fun close() {
        if (closed) return
        synchronized(this) {
            if (closed) return
            closed = true
            managed.asReversed().forEach { (session, options) ->
                runCatching { session.close() }
                runCatching { options.close() }
            }
        }
    }
}
