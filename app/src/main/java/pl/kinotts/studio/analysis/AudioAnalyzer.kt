package pl.kinotts.studio.analysis

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt
import kotlin.math.roundToInt

/** Audio decoded by Android's platform codecs; the source is never uploaded. */
data class AudioAnalysisResult(
    val wavFile: File,
    val durationSeconds: Double,
    val sampleRate: Int,
    val sourceChannels: Int,
    val peak: Float,
    val rmsDbfs: Float,
    val clippingPercent: Float,
    val silencePercent: Float,
    val monoSamples: Long
)

class AudioAnalyzer(context: Context) {
    private val appContext = context.applicationContext

    fun analyze(uri: Uri, onProgress: (Float) -> Unit = {}): AudioAnalysisResult {
        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        var decoderStarted = false
        val outDir = File(appContext.cacheDir, "voice-analysis")
        check(outDir.mkdirs() || outDir.isDirectory) { "Nie można utworzyć katalogu analizy" }
        val target = File(outDir, "reference_${System.currentTimeMillis()}.wav")
        var raf: RandomAccessFile? = null

        try {
            extractor.setDataSource(appContext, uri, null)
            var audioTrack = -1
            var inputFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) {
                    audioTrack = i
                    inputFormat = format
                    break
                }
            }
            require(audioTrack >= 0 && inputFormat != null) {
                "Wybrany plik nie zawiera ścieżki audio"
            }

            val format = inputFormat!!
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) {
                format.getLong(MediaFormat.KEY_DURATION).coerceAtLeast(0L)
            } else 0L
            extractor.selectTrack(audioTrack)
            decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(format, null, null, 0)
            decoder.start()
            decoderStarted = true

            var sampleRate = if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE)) format.getInteger(MediaFormat.KEY_SAMPLE_RATE) else 0
            var channels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 0
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
            var inputEnded = false
            var outputEnded = false
            var sampleCount = 0L
            var dataBytes = 0L
            var peak = 0f
            var squareSum = 0.0
            var silenceSamples = 0L
            var clippedSamples = 0L
            raf = RandomAccessFile(target, "rw")
            raf.write(ByteArray(44))

            val info = MediaCodec.BufferInfo()
            while (!outputEnded) {
                if (!inputEnded) {
                    val inputIndex = decoder.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputIndex)
                            ?: error("Brak bufora wejściowego dekodera")
                        inputBuffer.clear()
                        val size = extractor.readSampleData(inputBuffer, 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(inputIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEnded = true
                        } else {
                            val timeUs = extractor.sampleTime.coerceAtLeast(0L)
                            decoder.queueInputBuffer(inputIndex, 0, size, timeUs, extractor.sampleFlags)
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = decoder.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val decoded = decoder.outputFormat
                        if (decoded.containsKey(MediaFormat.KEY_SAMPLE_RATE)) sampleRate = decoded.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        if (decoded.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) channels = decoded.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        if (decoded.containsKey(MediaFormat.KEY_PCM_ENCODING)) pcmEncoding = decoded.getInteger(MediaFormat.KEY_PCM_ENCODING)
                    }
                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                    else -> if (outputIndex >= 0) {
                        if (info.size > 0) {
                            require(sampleRate > 0 && channels in 1..8) { "Dekoder zwrócił nieprawidłowy format audio" }
                            val bytesPerSample = when (pcmEncoding) {
                                AudioFormat.ENCODING_PCM_8BIT -> 1
                                AudioFormat.ENCODING_PCM_16BIT -> 2
                                AudioFormat.ENCODING_PCM_FLOAT -> 4
                                else -> error("Nieobsługiwany format PCM: $pcmEncoding")
                            }
                            val buffer = decoder.getOutputBuffer(outputIndex)
                                ?: error("Brak bufora wyjściowego dekodera")
                            val view = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                            view.position(info.offset)
                            view.limit(info.offset + info.size)
                            val raw = ByteArray(view.remaining())
                            view.get(raw)
                            val pcm = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
                            val frameBytes = bytesPerSample * channels
                            val frames = raw.size / frameBytes
                            val monoBytes = ByteArray(frames * 2)
                            var writeAt = 0

                            repeat(frames) {
                                var sum = 0f
                                repeat(channels) {
                                    val value = when (pcmEncoding) {
                                        AudioFormat.ENCODING_PCM_8BIT -> ((pcm.get().toInt() and 0xff) - 128) / 128f
                                        AudioFormat.ENCODING_PCM_16BIT -> pcm.short / 32768f
                                        AudioFormat.ENCODING_PCM_FLOAT -> pcm.float
                                        else -> 0f
                                    }
                                    sum += value.coerceIn(-1f, 1f)
                                }
                                val mono = (sum / channels).coerceIn(-1f, 1f)
                                val magnitude = abs(mono)
                                if (magnitude > peak) peak = magnitude
                                squareSum += mono.toDouble() * mono.toDouble()
                                if (magnitude < 0.01f) silenceSamples++
                                if (magnitude >= 0.999f) clippedSamples++
                                val quantized = (mono * 32767f).roundToInt().coerceIn(-32768, 32767)
                                monoBytes[writeAt++] = quantized.toByte()
                                monoBytes[writeAt++] = (quantized shr 8).toByte()
                            }
                            if (writeAt > 0) {
                                raf.write(monoBytes, 0, writeAt)
                                dataBytes += writeAt
                                sampleCount += frames
                            }
                            if (durationUs > 0L) {
                                onProgress((info.presentationTimeUs.toDouble() / durationUs).toFloat().coerceIn(0f, 0.99f))
                            }
                        }
                        if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outputEnded = true
                        decoder.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }

            check(dataBytes > 0L && sampleCount > 0L) { "Nie udało się odczytać próbek audio" }
            check(dataBytes <= Int.MAX_VALUE - 36L) { "Plik audio jest zbyt duży do zapisania jako WAV" }
            check(sampleRate > 0) { "Nieznana częstotliwość próbkowania" }
            raf.seek(0L)
            raf.write(wavHeader(dataBytes.toInt(), sampleRate))
            raf.fd.sync()
            raf.close()
            raf = null

            val rms = sqrt(squareSum / sampleCount.toDouble()).toFloat()
            val rmsDb = (20.0 * log10(rms.toDouble().coerceAtLeast(1e-9))).toFloat()
            onProgress(1f)
            return AudioAnalysisResult(
                wavFile = target,
                durationSeconds = sampleCount.toDouble() / sampleRate,
                sampleRate = sampleRate,
                sourceChannels = channels,
                peak = peak,
                rmsDbfs = rmsDb,
                clippingPercent = clippedSamples * 100f / sampleCount,
                silencePercent = silenceSamples * 100f / sampleCount,
                monoSamples = sampleCount
            )
        } catch (e: Exception) {
            target.delete()
            throw e
        } finally {
            try { raf?.close() } catch (_: Exception) { }
            if (decoderStarted) try { decoder?.stop() } catch (_: Exception) { }
            try { decoder?.release() } catch (_: Exception) { }
            try { extractor.release() } catch (_: Exception) { }
        }
    }

    private fun wavHeader(dataSize: Int, sampleRate: Int): ByteArray {
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + dataSize)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1)
        header.putShort(1)
        header.putInt(sampleRate)
        header.putInt(sampleRate * 2)
        header.putShort(2)
        header.putShort(16)
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataSize)
        return header.array()
    }
}
