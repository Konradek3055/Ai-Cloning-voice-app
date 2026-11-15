package pl.kinotts.studio.models

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.security.MessageDigest
import pl.kinotts.studio.storage.KinoStorage

/**
 * Authenticated in-app downloader for the INT8 XTTSv2 ONNX pack.
 *
 * Token jest przekazywany tylko do metody pobierania.
 * Nie jest zapisywany w SharedPreferences, plikach ani bazie danych.
 *
 * Model trafia do prywatnego katalogu filesDir aplikacji.
 */
class XttsModelManager(context: Context) {

    private val appContext = context.applicationContext
    private val root = KinoStorage.modelsRoot(appContext).apply { mkdirs() }

    private val staging =
        File(root, ".xttsv2-staging")

    private val installed = KinoStorage.modelsDir(appContext)

    private data class RemoteFile(
        val path: String,
        val size: Long,
        val sha256: String?
    )

    /**
     * Wymagane pliki pakietu XTTSv2.
     * Pobieramy GPT INT8 zamiast pełnej wersji FP32.
     */
    private val required = listOf(
        "xtts_onnx/conditioning_encoder.onnx",
        "xtts_onnx/speaker_encoder.onnx",
        "xtts_onnx/gpt_model_int8.onnx",
        "xtts_onnx/hifigan_vocoder.onnx",
        "xtts_onnx/metadata.json",
        "xtts_onnx/vocab.json",
        "xtts_onnx/mel_stats.npy",
        "xtts_onnx/embeddings/mel_embedding.npy",
        "xtts_onnx/embeddings/mel_pos_embedding.npy",
        "xtts_onnx/embeddings/text_embedding.npy",
        "xtts_onnx/embeddings/text_pos_embedding.npy"
    )

    fun modelDir(): File = installed

    /**
     * Sprawdza, czy wszystkie wymagane pliki są zainstalowane.
     */
    fun isInstalled(): Boolean {
        return required.all { remote ->
            val relative = remote.removePrefix("xtts_onnx/")
            val file = File(installed, relative)

            file.isFile && file.length() > 0L
        }
    }

    /**
     * Pobiera model z Hugging Face i instaluje go atomowo.
     */
    suspend fun downloadAndInstall(
        token: String,
        onProgress: (Float) -> Unit = {}
    ): File = withContext(Dispatchers.IO) {

        val coroutineContext = currentCoroutineContext()

        KinoStorage.ensureDirectories(appContext)

        val cleanToken = token.trim()

        require(
            cleanToken.startsWith("hf_") &&
            cleanToken.length >= 20
        ) {
            "Wpisz prawidłowy token Hugging Face zaczynający się od hf_."
        }

        check(root.mkdirs() || root.isDirectory) {
            "Nie można utworzyć prywatnego katalogu modeli."
        }

        onProgress(0f)

        // Lista plików dostępnych dla konta użytkownika.
        val remoteFiles = listRemoteFiles(cleanToken)
            .associateBy { it.path }

        // Sprawdzenie kompletności repozytorium.
        val missing = required.filterNot {
            remoteFiles.containsKey(it)
        }

        require(missing.isEmpty()) {
            "Brak wymaganych plików: ${missing.joinToString()}"
        }

        val selected = required.map {
            remoteFiles.getValue(it)
        }

        require(selected.all { it.size > 0L }) {
            "Hugging Face nie zwrócił rozmiarów wszystkich plików."
        }

        val totalBytes = selected.sumOf { it.size }

        require(
            totalBytes in 100_000_000L..3_000_000_000L
        ) {
            "Nieprawidłowy rozmiar pakietu XTTS: $totalBytes bajtów."
        }

        // Kontrola dostępnego miejsca.
        val available = StatFs(root.absolutePath).availableBytes

        require(
            available >= totalBytes + 128L * 1024L * 1024L
        ) {
            "Za mało wolnego miejsca. Model wymaga około " +
                "${totalBytes / (1024 * 1024)} MB plus zapas."
        }

        // Przygotowanie katalogu instalacyjnego.
        staging.deleteRecursively()

        check(staging.mkdirs() || staging.isDirectory) {
            "Nie można utworzyć katalogu tymczasowego XTTS."
        }

        var completedBytes = 0L

        try {

            for (remote in selected) {

                coroutineContext.ensureActive()

                val relative =
                    remote.path.removePrefix("xtts_onnx/")

                require(
                    relative.isNotBlank() &&
                    !relative.startsWith("/") &&
                    !relative.contains("..")
                ) {
                    "Nieprawidłowa ścieżka pliku modelu."
                }

                val target = File(
                    staging,
                    relative
                ).canonicalFile

                val stagingCanonical = staging.canonicalFile

                require(
                    target.path.startsWith(
                        stagingCanonical.path + File.separator
                    )
                ) {
                    "Plik modelu wychodzi poza katalog instalacyjny."
                }

                target.parentFile?.mkdirs()

                val partial = File(
                    target.parentFile,
                    target.name + ".part"
                )

                partial.delete()

                val encodedPath = remote.path
                    .split("/")
                    .joinToString("/") {
                        URLEncoder.encode(it, "UTF-8")
                            .replace("+", "%20")
                    }

                val address =
                    "https://huggingface.co/$MODEL_ID/resolve/main/" +
                    "$encodedPath?download=true"

                val connection =
                    openConnection(address, cleanToken)

                try {

                    val response = connection.responseCode

                    if (response !in 200..299) {

                        val reason = when (response) {

                            401 ->
                                "Token Hugging Face został odrzucony."

                            403 ->
                                "Brak dostępu do modelu albo uprawnienia Read."

                            404 ->
                                "Nie znaleziono pliku: ${remote.path}"

                            else ->
                                "Błąd pobierania ${remote.path}: HTTP $response."
                        }

                        throw IOException(reason)
                    }

                    val digest =
                        MessageDigest.getInstance("SHA-256")

                    var fileBytes = 0L

                    connection.inputStream.use { raw ->

                        BufferedInputStream(raw).use { input ->

                            FileOutputStream(partial).use { output ->

                                val buffer = ByteArray(1024 * 1024)

                                while (true) {

                                    coroutineContext.ensureActive()

                                    val count = input.read(buffer)

                                    if (count < 0) break
                                    if (count == 0) continue

                                    output.write(buffer, 0, count)

                                    digest.update(
                                        buffer,
                                        0,
                                        count
                                    )

                                    fileBytes += count

                                    val overall =
                                        completedBytes + fileBytes

                                    val value =
                                        overall.toDouble() /
                                        totalBytes.toDouble()

                                    onProgress(
                                        value.toFloat()
                                            .coerceIn(0f, 0.995f)
                                    )
                                }

                                output.fd.sync()
                            }
                        }
                    }

                    // Kontrola rozmiaru pliku.
                    require(fileBytes == remote.size) {
                        "Niepełny plik ${remote.path}: " +
                        "$fileBytes z ${remote.size} bajtów."
                    }

                    // Kontrola SHA-256, jeśli Hugging Face udostępnia hash.
                    val actualHash = digest.digest()
                        .joinToString("") {
                            "%02x".format(it)
                        }

                    if (!remote.sha256.isNullOrBlank()) {

                        require(
                            actualHash.equals(
                                remote.sha256,
                                ignoreCase = true
                            )
                        ) {
                            "Nieprawidłowy SHA-256 pliku ${remote.path}."
                        }
                    }

                    check(partial.renameTo(target)) {
                        "Nie udało się zapisać pliku ${remote.path}."
                    }

                    completedBytes += fileBytes

                } finally {

                    connection.disconnect()

                    if (partial.exists()) {
                        partial.delete()
                    }
                }
            }

            // Kontrola kompletności całego pakietu.
            val complete = required.all { remote ->

                val file = File(
                    staging,
                    remote.removePrefix("xtts_onnx/")
                )

                file.isFile && file.length() > 0L
            }

            check(complete) {
                "Pakiet XTTS nie przeszedł kontroli kompletności."
            }

            // Instalacja z możliwością przywrócenia poprzedniej wersji.
            val old = File(root, "xttsv2-old")

            old.deleteRecursively()

            if (installed.exists()) {

                check(installed.renameTo(old)) {
                    "Nie można odłożyć poprzedniej instalacji XTTS."
                }
            }

            if (!staging.renameTo(installed)) {

                if (old.exists()) {
                    old.renameTo(installed)
                }

                error(
                    "Nie udało się zainstalować modelu XTTS."
                )
            }

            old.deleteRecursively()

            check(isInstalled()) {
                "Model XTTS nie przeszedł końcowej kontroli."
            }

            onProgress(1f)

            installed

        } catch (error: Throwable) {

            staging.deleteRecursively()

            throw error
        }
    }

    /**
     * Pobiera listę plików z API Hugging Face.
     */
    private fun listRemoteFiles(
        token: String
    ): List<RemoteFile> {

        val address =
            "https://huggingface.co/api/models/$MODEL_ID/" +
            "tree/main?recursive=true&expand=true"

        val connection =
            openConnection(address, token)

        try {

            val response = connection.responseCode

            if (response !in 200..299) {

                val message = when (response) {

                    401 ->
                        "Hugging Face odrzucił token. Sprawdź uprawnienie Read."

                    403 ->
                        "Konto nie ma dostępu do modelu albo token nie ma uprawnienia Read."

                    else ->
                        "Nie udało się pobrać listy plików: HTTP $response."
                }

                throw IOException(message)
            }

            val json = connection.inputStream
                .bufferedReader(Charsets.UTF_8)
                .use { it.readText() }

            val array = JSONArray(json)

            val result =
                ArrayList<RemoteFile>(array.length())

            for (index in 0 until array.length()) {

                val item =
                    array.optJSONObject(index) ?: continue

                if (item.optString("type") != "file") {
                    continue
                }

                val path = item.optString("path")

                if (path !in required) {
                    continue
                }

                val lfs = item.optJSONObject("lfs")

                val rawHash =
                    lfs?.optString("sha256")
                        ?.takeIf {
                            it.isNotBlank() && it != "null"
                        }
                    ?: lfs?.optString("oid")
                        ?.takeIf {
                            it.isNotBlank() && it != "null"
                        }

                val hash = rawHash
                    ?.removePrefix("sha256:")
                    ?.takeIf {
                        it.matches(
                            Regex("[0-9a-fA-F]{64}")
                        )
                    }

                val lfsSize = lfs?.optLong("size", -1L) ?: -1L
                val apiSize = item.optLong("size", -1L)
                val size = if (lfsSize > 0L) lfsSize else apiSize

                result += RemoteFile(
                    path = path,
                    size = size,
                    sha256 = hash
                )
            }

            return result

        } finally {

            connection.disconnect()
        }
    }

    private fun openConnection(
        address: String,
        token: String
    ): HttpURLConnection {

        return (URL(address).openConnection()
            as HttpURLConnection).apply {

            connectTimeout = 30_000
            readTimeout = 180_000

            instanceFollowRedirects = true

            requestMethod = "GET"

            setRequestProperty(
                "Authorization",
                "Bearer $token"
            )

            setRequestProperty(
                "User-Agent",
                "KinoTTS-Studio-Android"
            )

            setRequestProperty(
                "Cache-Control",
                "no-cache"
            )
        }
    }

    companion object {

        private const val MODEL_ID =
            "pltobing/XTTSv2-Streaming-ONNX"
    }
}
