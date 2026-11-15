package pl.kinotts.studio.xtts

import java.io.Closeable

/** JNI bridge to Hugging Face's official Rust Tokenizers implementation. */
internal class HFTokenizer(tokenizerBytes: ByteArray) : Closeable {
    private var tokenizerPtr: Long = 0L

    init {
        tokenizerPtr = nativeCreateTokenizer(tokenizerBytes)
        check(tokenizerPtr != 0L) {
            "Hugging Face Tokenizers nie może odczytać tokenizer JSON z vocab.json."
        }
    }

    @Synchronized
    fun encode(text: String): IntArray {
        check(tokenizerPtr != 0L) { "Tokenizer Hugging Face został zamknięty." }
        return nativeEncode(tokenizerPtr, text)
            ?: error("Hugging Face Tokenizers nie zdołał zakodować tekstu.")
    }

    @Synchronized
    fun tokenId(token: String): Int {
        check(tokenizerPtr != 0L) { "Tokenizer Hugging Face został zamknięty." }
        return nativeTokenId(tokenizerPtr, token)
    }

    @Synchronized
    override fun close() {
        val ptr = tokenizerPtr
        if (ptr != 0L) {
            tokenizerPtr = 0L
            nativeDeleteTokenizer(ptr)
        }
    }

    private external fun nativeCreateTokenizer(tokenizerBytes: ByteArray): Long
    private external fun nativeEncode(tokenizerPtr: Long, text: String): IntArray?
    private external fun nativeTokenId(tokenizerPtr: Long, token: String): Int
    private external fun nativeDeleteTokenizer(tokenizerPtr: Long)

    companion object {
        init {
            System.loadLibrary("kinotts_tokenizer")
        }
    }
}
