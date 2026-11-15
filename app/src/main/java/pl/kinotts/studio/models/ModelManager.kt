package pl.kinotts.studio.models

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/** Downloads and stores model packs inside this app's private files directory. */
class ModelManager(context: Context) {
    private val root = File(context.filesDir, "models").apply { mkdirs() }
    private val supertonicDir = File(root, "supertonic3")
    private val supertonicUrl =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/" +
        "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"

    fun isSupertonicInstalled(): Boolean = requiredSupertonicFiles().all { File(supertonicDir, it).isFile }

    suspend fun ensureSupertonic(onProgress: (Float) -> Unit = {}): File = withContext(Dispatchers.IO) {
        if (isSupertonicInstalled()) return@withContext findModelRoot()
        val archive = File(root, "supertonic3.tar.bz2.part")
        val connection = (URL(supertonicUrl).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            requestMethod = "GET"
        }
        try {
            connection.connect()
            if (connection.responseCode !in 200..299) error("HTTP ${connection.responseCode}")
            val total = connection.contentLengthLong
            connection.inputStream.use { input ->
                FileOutputStream(archive).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    var read: Int
                    var done = 0L
                    while (input.read(buffer).also { read = it } >= 0) {
                        if (read == 0) continue
                        output.write(buffer, 0, read)
                        done += read
                        if (total > 0) onProgress((done.toFloat() / total).coerceIn(0f, 1f))
                    }
                    output.fd.sync()
                }
            }
        } finally { connection.disconnect() }

        val staging = File(root, "supertonic3-staging").apply { deleteRecursively(); mkdirs() }
        try {
            BZip2CompressorInputStream(BufferedInputStream(archive.inputStream())).use { bz ->
                TarArchiveInputStream(bz).use { tar ->
                    while (true) {
                        val entry = tar.nextTarEntry ?: break
                        val out = File(staging, entry.name).canonicalFile
                        require(out.path.startsWith(staging.canonicalPath + File.separator) || out == staging) {
                            "Nieprawidłowa ścieżka w archiwum"
                        }
                        if (entry.isDirectory) out.mkdirs() else {
                            out.parentFile?.mkdirs()
                            FileOutputStream(out).use { stream -> tar.copyTo(stream) }
                        }
                    }
                }
            }
            val extractedRoot = findContainingModel(staging)
                ?: error("Archiwum nie zawiera kompletu modelu Supertonic 3")
            supertonicDir.deleteRecursively()
            if (extractedRoot == staging) {
                require(staging.renameTo(supertonicDir)) { "Nie udało się przenieść modelu" }
            } else {
                require(extractedRoot.copyRecursively(supertonicDir, overwrite = true).exists())
            }
            check(isSupertonicInstalled()) { "Niekompletny model po rozpakowaniu" }
            onProgress(1f)
            findModelRoot()
        } finally {
            archive.delete()
            staging.deleteRecursively()
        }
    }

    private fun findModelRoot(): File {
        val nested = findContainingModel(supertonicDir)
        return nested ?: error("Brak wymaganych plików modelu Supertonic 3")
    }

    private fun findContainingModel(base: File): File? {
        if (requiredSupertonicFiles().all { File(base, it).isFile }) return base
        base.walkTopDown().filter { it.isDirectory }.forEach { dir ->
            if (requiredSupertonicFiles().all { File(dir, it).isFile }) return dir
        }
        return null
    }

    private fun requiredSupertonicFiles() = listOf(
        "duration_predictor.int8.onnx", "text_encoder.int8.onnx",
        "vector_estimator.int8.onnx", "vocoder.int8.onnx", "tts.json",
        "unicode_indexer.bin", "voice.bin"
    )
}
