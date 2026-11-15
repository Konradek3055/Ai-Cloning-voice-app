package pl.kinotts.studio.voices

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import pl.kinotts.studio.analysis.AudioAnalysisResult
import pl.kinotts.studio.storage.KinoStorage
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import kotlin.math.ceil

/** One recorded sample. Several samples can share groupId and therefore one voice. */
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
    val language: String,
    val groupId: String = id
)

class VoiceProfileStore(context: Context) {
    private val appContext = context.applicationContext
    private val root = KinoStorage.voicesDir(appContext).apply { mkdirs() }
    private val index = File(root, "profiles.json")
    private val compositeDir = File(appContext.cacheDir, "kinotts-combined-voices")

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

    /** Save a new voice (groupId=null) or append a new sample to an existing voice. */
    @Synchronized
    fun save(
        name: String,
        analysis: AudioAnalysisResult,
        language: String,
        consentConfirmed: Boolean,
        groupId: String? = null
    ): VoiceProfile {
        require(consentConfirmed) { "Potwierdź prawo do użycia nagrania" }
        KinoStorage.ensureDirectories(appContext)
        require(analysis.wavFile.isFile && analysis.wavFile.length() > 44L) { "Brak próbki WAV do zapisania" }

        val allExisting = list()
        val existingGroup = groupId?.let { target -> allExisting.filter { it.groupId == target } }
        if (groupId != null) require(!existingGroup.isNullOrEmpty()) { "Wybrany profil głosu już nie istnieje" }
        val cleanName = if (!existingGroup.isNullOrEmpty()) existingGroup.first().name
            else name.trim().replace(Regex("\\s+"), " ").also {
                require(it.length in 2..40) { "Nazwa profilu musi mieć 2–40 znaków" }
            }
        val actualGroupId = existingGroup?.firstOrNull()?.groupId ?: UUID.randomUUID().toString()

        check(root.mkdirs() || root.isDirectory) { "Nie można utworzyć katalogu profili" }
        val id = UUID.randomUUID().toString()
        val destination = File(root, "$id.wav")
        val partial = File(root, "$id.wav.part")
        analysis.wavFile.copyTo(partial, overwrite = true)
        check(partial.renameTo(destination)) { "Nie udało się zapisać pliku próbki" }

        val profile = VoiceProfile(
            id = id,
            groupId = actualGroupId,
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
            writeIndex(allExisting + profile)
        } catch (e: Exception) {
            destination.delete()
            throw e
        }
        invalidateComposite(actualGroupId)
        return profile
    }

    @Synchronized
    fun delete(id: String): Boolean {
        require(runCatching { UUID.fromString(id) }.isSuccess) { "Nieprawidłowy identyfikator próbki" }
        val all = list()
        val sample = all.firstOrNull { it.id == id } ?: return false
        val rootCanonical = root.canonicalFile
        val wav = File(sample.wavPath).canonicalFile
        require(wav.parentFile == rootCanonical && wav.extension.equals("wav", true)) {
            "Plik próbki znajduje się poza katalogiem profili"
        }
        writeIndex(all.filterNot { it.id == id })
        if (wav.exists() && !wav.delete()) return false
        invalidateComposite(sample.groupId)
        return true
    }

    /**
     * Join all samples belonging to a voice into one mono PCM16 WAV for XTTS.
     * XTTS currently accepts one reference WAV, so samples are resampled to 22.05 kHz,
     * concatenated with short gaps, and capped at six seconds (the XTTS frontend cap).
     */
    @Synchronized
    fun combinedReference(groupId: String): File {
        val samples = list().filter { it.groupId == groupId }.sortedBy { it.createdAt }
        require(samples.isNotEmpty()) { "Profil głosu nie zawiera żadnych próbek" }
        val signature = samples.joinToString("|") {
            val f = File(it.wavPath)
            "${it.id}:${f.lastModified()}:${f.length()}"
        }
        val key = Integer.toHexString(signature.hashCode())
        check(compositeDir.mkdirs() || compositeDir.isDirectory) { "Nie można utworzyć cache próbek głosu" }
        val target = File(compositeDir, "voice_${groupId}_$key.wav")
        if (target.isFile && target.length() > 44L) return target

        val targetRate = 22050
        val maxFrames = targetRate * 6
        val gapFrames = minOf(targetRate / 5, maxFrames / (samples.size * 2).coerceAtLeast(1))
        val pcm = ByteArrayOutputStream(maxFrames * 2)
        var framesWritten = 0
        for ((sampleIndex, sample) in samples.withIndex()) {
            if (framesWritten >= maxFrames) break
            val wav = readPcm16Mono(File(sample.wavPath))
            val ratio = targetRate.toDouble() / wav.sampleRate.toDouble()
            val outputFrames = ceil(wav.samples.size * ratio).toInt()
            val remainingSamples = samples.size - sampleIndex
            val reserveForGaps = gapFrames * (remainingSamples - 1).coerceAtLeast(0)
            val fairBudget = (maxFrames - framesWritten - reserveForGaps).coerceAtLeast(0) / remainingSamples
            val count = minOf(outputFrames, fairBudget, maxFrames - framesWritten)
            val bytes = ByteBuffer.allocate(count * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until count) {
                val sourcePosition = i / ratio
                val left = sourcePosition.toInt().coerceIn(0, wav.samples.lastIndex)
                val right = (left + 1).coerceAtMost(wav.samples.lastIndex)
                val fraction = (sourcePosition - left).toFloat()
                val value = wav.samples[left] + (wav.samples[right] - wav.samples[left]) * fraction
                bytes.putShort(value.toInt().coerceIn(-32768, 32767).toShort())
            }
            pcm.write(bytes.array())
            framesWritten += count
            if (framesWritten < maxFrames && sampleIndex < samples.lastIndex) {
                val gap = minOf(gapFrames, maxFrames - framesWritten)
                pcm.write(ByteArray(gap * 2))
                framesWritten += gap
            }
        }
        require(framesWritten > 0) { "Nie udało się połączyć próbek profilu" }
        val data = pcm.toByteArray()
        val tmp = File(target.parentFile, target.name + ".part")
        tmp.outputStream().use { out ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray(Charsets.US_ASCII)); header.putInt(36 + data.size)
            header.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)); header.putInt(16)
            header.putShort(1); header.putShort(1); header.putInt(targetRate)
            header.putInt(targetRate * 2); header.putShort(2); header.putShort(16)
            header.put("data".toByteArray(Charsets.US_ASCII)); header.putInt(data.size)
            out.write(header.array()); out.write(data)
        }
        check(tmp.renameTo(target) || (tmp.copyTo(target, overwrite = true).exists() && tmp.delete())) {
            "Nie udało się zapisać połączonej próbki profilu"
        }
        invalidateComposite(groupId, keep = target.name)
        return target
    }

    private data class PcmWav(val sampleRate: Int, val samples: ShortArray)

    private fun readPcm16Mono(file: File): PcmWav {
        val bytes = file.readBytes()
        require(bytes.size >= 44 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
            String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") { "Nieprawidłowy WAV próbki: ${file.name}" }
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12; var format = 0; var channels = 0; var rate = 0; var bits = 0
        var dataOffset = -1; var dataLength = 0
        while (pos + 8 <= bytes.size) {
            val size = b.getInt(pos + 4).toLong() and 0xffffffffL
            val body = pos + 8
            if (size > Int.MAX_VALUE || body.toLong() + size > bytes.size) break
            when (String(bytes, pos, 4, Charsets.US_ASCII)) {
                "fmt " -> if (size >= 16) {
                    format = b.getShort(body).toInt() and 0xffff
                    channels = b.getShort(body + 2).toInt() and 0xffff
                    rate = b.getInt(body + 4)
                    bits = b.getShort(body + 14).toInt() and 0xffff
                }
                "data" -> { dataOffset = body; dataLength = size.toInt() }
            }
            pos = body + size.toInt() + (size.toInt() and 1)
        }
        require(format == 1 && channels == 1 && bits == 16 && rate > 0 && dataOffset >= 0) {
            "Próbka musi być WAV PCM16 mono: ${file.name}"
        }
        val count = dataLength / 2
        val samples = ShortArray(count)
        val view = ByteBuffer.wrap(bytes, dataOffset, count * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in samples.indices) samples[i] = view.short
        require(samples.isNotEmpty()) { "Pusta próbka: ${file.name}" }
        return PcmWav(rate, samples)
    }

    private fun writeIndex(all: List<VoiceProfile>) {
        val json = JSONArray().apply { all.forEach { put(it.toJson()) } }
        val tmp = File(root, "profiles.json.part")
        tmp.writeText(json.toString(2), Charsets.UTF_8)
        check(tmp.renameTo(index) || (tmp.copyTo(index, overwrite = true).exists() && tmp.delete())) {
            "Nie udało się zapisać indeksu profili"
        }
    }

    private fun invalidateComposite(groupId: String, keep: String? = null) {
        compositeDir.listFiles()?.filter {
            it.name.startsWith("voice_${groupId}_") && it.name != keep
        }?.forEach { it.delete() }
    }

    private fun VoiceProfile.toJson() = JSONObject().apply {
        put("id", id); put("groupId", groupId); put("name", name); put("wavPath", wavPath)
        put("createdAt", createdAt); put("sampleRate", sampleRate); put("durationSeconds", durationSeconds)
        put("rmsDbfs", rmsDbfs.toDouble()); put("peak", peak.toDouble())
        put("clippingPercent", clippingPercent.toDouble()); put("silencePercent", silencePercent.toDouble())
        put("language", language)
    }

    private fun JSONObject.toProfile(): VoiceProfile? = try {
        val id = getString("id")
        VoiceProfile(
            id = id,
            groupId = optString("groupId", id).ifBlank { id },
            name = getString("name"), wavPath = getString("wavPath"),
            createdAt = getLong("createdAt"), sampleRate = getInt("sampleRate"),
            durationSeconds = getDouble("durationSeconds"), rmsDbfs = getDouble("rmsDbfs").toFloat(),
            peak = getDouble("peak").toFloat(), clippingPercent = getDouble("clippingPercent").toFloat(),
            silencePercent = getDouble("silencePercent").toFloat(), language = optString("language", "pl")
        )
    } catch (_: Exception) { null }
}
