package pl.kinotts.studio.models

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/** Downloads, validates and installs model packs inside this app's private files directory. */
class ModelManager(context: Context) {
    private val root = File(context.filesDir, "models").apply { mkdirs() }
    private val supertonicDir = File(root, "supertonic3")
    private val supertonicUrl =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/" +
            "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"

    private val requiredFiles = listOf(
        "duration_predictor.int8.onnx", "text_encoder.int8.onnx",
        "vector_estimator.int8.onnx", "vocoder.int8.onnx", "tts.json",
        "unicode_indexer.bin", "voice.bin"
    )

    /** Checks the discovered model root, including archives that contain a nested folder. */
    fun isSupertonicInstalled(): Boolean = findContainingModel(supertonicDir) != null

    suspend fun ensureSupertonic(onProgress: (Float) -> Unit = {}): File =
        withContext(Dispatchers.IO) {
            if (isSupertonicInstalled()) return@withContext findModelRoot()

            check(root.mkdirs() || root.isDirectory) { "Nie można utworzyć katalogu modeli" }
            val archive = File(root, "supertonic3.tar.bz2.part")
            val staging = File(root, "supertonic3-staging")
            val installing = File(root, "supertonic3-installing")
            archive.delete()
            staging.deleteRecursively()
            installing.deleteRecursively()
            staging.mkdirs()

            val connection = (URL(supertonicUrl).openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                requestMethod = "GET"
                setRequestProperty("User-Agent", "KinoTTS-Studio-Android")
            }

            try {
                connection.connect()
                val response = connection.responseCode
                if (response !in 200..299) error("Pobieranie modelu nie powiodło się: HTTP $response")

                val total = connection.contentLengthLong
                connection.inputStream.use { input ->
                    FileOutputStream(archive).use { output ->
                        val buffer = ByteArray(256 * 1024)
                        var done = 0L
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            if (count == 0) continue
                            output.write(buffer, 0, count)
                            done += count
                            if (total > 0L) onProgress((done.toDouble() / total).toFloat().coerceIn(0f, 0.99f))
                        }
                        output.fd.sync()
                    }
                }
                check(archive.isFile && archive.length() > 1024L) { "Pobrane archiwum jest puste lub uszkodzone" }
            } catch (e: Exception) {
                archive.delete()
                staging.deleteRecursively()
                installing.deleteRecursively()
                throw e
            } finally {
                connection.disconnect()
            }

            try {
                BZip2CompressorInputStream(BufferedInputStream(archive.inputStream())).use { bz ->
                    TarArchiveInputStream(bz).use { tar ->
                        while (true) {
                            val entry = tar.nextTarEntry ?: break
                            val out = File(staging, entry.name).canonicalFile
                            val stagingPath = staging.canonicalPath
                            require(out.path == stagingPath || out.path.startsWith(stagingPath + File.separator)) {
                                "Nieprawidłowa ścieżka w archiwum"
                            }
                            when {
                                entry.isDirectory -> out.mkdirs()
                                entry.isFile -> {
                                    out.parentFile?.mkdirs()
                                    FileOutputStream(out).use { stream -> tar.copyTo(stream) }
                                }
                                else -> error("Archiwum zawiera niedozwolony typ wpisu: ${entry.name}")
                            }
                        }
                    }
                }

                val extractedRoot = findContainingModel(staging)
                    ?: error("Archiwum nie zawiera kompletnego modelu Supertonic 3")
                check(extractedRoot.copyRecursively(installing, overwrite = true)) {
                    "Nie udało się przygotować modelu do instalacji"
                }
                check(findContainingModel(installing) != null) { "Model po rozpakowaniu jest niekompletny" }

                val old = File(root, "supertonic3-old")
                old.deleteRecursively()
                if (supertonicDir.exists()) {
                    check(supertonicDir.renameTo(old)) { "Nie można odłożyć poprzedniego modelu" }
                }
                if (!installing.renameTo(supertonicDir)) {
                    if (old.exists()) old.renameTo(supertonicDir)
                    error("Nie udało się zainstalować modelu Supertonic 3")
                }
                old.deleteRecursively()
                check(isSupertonicInstalled()) { "Model nie przeszedł końcowej weryfikacji" }
                onProgress(1f)
                findModelRoot()
            } finally {
                archive.delete()
                staging.deleteRecursively()
                installing.deleteRecursively()
            }
        }

    private fun findModelRoot(): File =
        findContainingModel(supertonicDir) ?: error("Brak wymaganych plików modelu Supertonic 3")

    private fun findContainingModel(base: File): File? {
        if (!base.isDirectory) return null
        if (requiredFiles.all { File(base, it).isFile && File(base, it).length() > 0L }) return base
        base.walkTopDown().filter { it.isDirectory }.forEach { dir ->
            if (requiredFiles.all { File(dir, it).isFile && File(dir, it).length() > 0L }) return dir
        }
        return null
    }
}
