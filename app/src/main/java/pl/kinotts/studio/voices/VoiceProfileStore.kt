package pl.kinotts.studio.voices

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import pl.kinotts.studio.analysis.AudioAnalysisResult
import pl.kinotts.studio.storage.KinoStorage
import java.io.File
import java.util.UUID

/** Stores consent-confirmed reference samples and their analysis in app-private storage. */
data class VoiceProfile(
    val id: String,
    val name: String,
    val wavPath: String,
    val createdAt: Long,
    val sampleRate: Int,
    val durationSeconds: Double,
    val rmsDbfs: Float,
    val peak: Float,
    val clippingPercent: Float,
    val silencePercent: Float,
    val language: String
)

class VoiceProfileStore(context: Context) {
    private val appContext = context.applicationContext
    private val root = KinoStorage.voicesDir(appContext).apply { mkdirs() }
    private val index = File(root, "profiles.json")

    @Synchronized
    fun list(): List<VoiceProfile> {
        if (!index.isFile) return emptyList()
        return try {
            val array = JSONArray(index.readText(Charsets.UTF_8))
            buildList {
                for (i in 0 until array.length()) {
                    val item = array.optJSONObject(i) ?: continue
                    val profile = item.toProfile() ?: continue
                    if (File(profile.wavPath).isFile) add(profile)
                }
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    @Synchronized
    fun save(
        name: String,
        analysis: AudioAnalysisResult,
        language: String,
        consentConfirmed: Boolean
    ): VoiceProfile {
        require(consentConfirmed) { "Potwierdź prawo do użycia nagrania" }
        KinoStorage.ensureDirectories(appContext)
        val cleanName = name.trim().replace(Regex("\\s+"), " ")
        require(cleanName.length in 2..40) { "Nazwa profilu musi mieć 2–40 znaków" }
        require(analysis.wavFile.isFile && analysis.wavFile.length() > 44L) { "Brak próbki WAV do zapisania" }

        check(root.mkdirs() || root.isDirectory) { "Nie można utworzyć katalogu profili" }
        val id = UUID.randomUUID().toString()
        val destination = File(root, "$id.wav")
        val partial = File(root, "$id.wav.part")
        analysis.wavFile.copyTo(partial, overwrite = true)
        check(partial.renameTo(destination)) { "Nie udało się zapisać pliku próbki" }

        val profile = VoiceProfile(
            id = id,
            name = cleanName,
            wavPath = destination.absolutePath,
            createdAt = System.currentTimeMillis(),
            sampleRate = analysis.sampleRate,
            durationSeconds = analysis.durationSeconds,
            rmsDbfs = analysis.rmsDbfs,
            peak = analysis.peak,
            clippingPercent = analysis.clippingPercent,
            silencePercent = analysis.silencePercent,
            language = language
        )

        try {
            val all = JSONArray()
            list().forEach { all.put(it.toJson()) }
            all.put(profile.toJson())
            val tmpIndex = File(root, "profiles.json.part")
            tmpIndex.writeText(all.toString(2), Charsets.UTF_8)
            check(tmpIndex.renameTo(index) || (tmpIndex.copyTo(index, overwrite = true).exists() && tmpIndex.delete())) {
                "Nie udało się zapisać indeksu profili"
            }
        } catch (e: Exception) {
            destination.delete()
            throw e
        }
        return profile
    }

    @Synchronized
    fun delete(id: String): Boolean {
        require(runCatching { UUID.fromString(id) }.isSuccess) { "Nieprawidłowy identyfikator profilu" }
        val profile = list().firstOrNull { it.id == id } ?: return false
        val rootCanonical = root.canonicalFile
        val wav = File(profile.wavPath).canonicalFile
        require(wav.parentFile == rootCanonical && wav.extension.equals("wav", true)) {
            "Plik próbki znajduje się poza katalogiem profili"
        }
        val remaining = list().filterNot { it.id == id }
        val tmp = File(root, "profiles.json.part")
        val json = JSONArray().apply { remaining.forEach { put(it.toJson()) } }
        tmp.writeText(json.toString(2), Charsets.UTF_8)
        check(tmp.renameTo(index) || (tmp.copyTo(index, overwrite = true).exists() && tmp.delete())) {
            "Nie udało się zaktualizować listy profili"
        }
        if (wav.exists() && !wav.delete()) return false
        return true
    }

    private fun VoiceProfile.toJson() = JSONObject().apply {
        put("id", id)
        put("name", name)
        put("wavPath", wavPath)
        put("createdAt", createdAt)
        put("sampleRate", sampleRate)
        put("durationSeconds", durationSeconds)
        put("rmsDbfs", rmsDbfs.toDouble())
        put("peak", peak.toDouble())
        put("clippingPercent", clippingPercent.toDouble())
        put("silencePercent", silencePercent.toDouble())
        put("language", language)
    }

    private fun JSONObject.toProfile(): VoiceProfile? = try {
        VoiceProfile(
            id = getString("id"),
            name = getString("name"),
            wavPath = getString("wavPath"),
            createdAt = getLong("createdAt"),
            sampleRate = getInt("sampleRate"),
            durationSeconds = getDouble("durationSeconds"),
            rmsDbfs = getDouble("rmsDbfs").toFloat(),
            peak = getDouble("peak").toFloat(),
            clippingPercent = getDouble("clippingPercent").toFloat(),
            silencePercent = getDouble("silencePercent").toFloat(),
            language = optString("language", "pl")
        )
    } catch (_: Exception) {
        null
    }
}
