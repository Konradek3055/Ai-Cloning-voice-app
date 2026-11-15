package pl.kinotts.studio.storage

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import java.io.File

/** Public, uninstall-persistent storage for user-owned KinoTTS assets. */
object KinoStorage {
    private fun documentsDir(): File =
        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)

    fun root(context: Context): File = File(documentsDir(), "KinoTTS")
    fun modelsRoot(context: Context): File = File(root(context), "Models")
    fun modelsDir(context: Context): File = File(modelsRoot(context), "XTTSv2")
    fun supertonicDir(context: Context): File = File(modelsRoot(context), "Supertonic3")
    fun voicesDir(context: Context): File = File(root(context), "Voices")
    fun outputDir(context: Context): File = File(root(context), "Output")

    fun hasAccess(context: Context): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            context.checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED
        }

    fun requireAccess(context: Context) {
        check(hasAccess(context)) {
            "Brak dostępu do pamięci współdzielonej. Otwórz Ustawienia → Dostęp do wszystkich plików dla KinoTTS i włącz dostęp."
        }
    }

    fun requestAccess(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val appIntent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                data = Uri.parse("package:${context.packageName}")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(appIntent)
            } catch (_: Exception) {
                context.startActivity(
                    Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                )
            }
        }
    }

    fun ensureDirectories(context: Context) {
        requireAccess(context)
        listOf(root(context), modelsRoot(context), modelsDir(context), supertonicDir(context), voicesDir(context), outputDir(context))
            .forEach { dir -> check(dir.mkdirs() || dir.isDirectory) { "Nie można utworzyć katalogu: ${dir.absolutePath}" } }
        migrateLegacy(context)
    }

    /** Copy old app-private data without deleting the source. Safe to repeat. */
    fun migrateLegacy(context: Context) {
        if (!hasAccess(context)) return
        val files = context.applicationContext.filesDir
        val oldModel = File(files, "models/xttsv2")
        val newModel = modelsDir(context)
        if (oldModel.isDirectory) {
            check(newModel.mkdirs() || newModel.isDirectory) { "Nie można utworzyć katalogu modelu XTTS." }
            oldModel.walkTopDown().filter { it.isFile }.forEach { source ->
                val target = File(newModel, source.relativeTo(oldModel).path)
                if (!target.exists()) {
                    target.parentFile?.mkdirs()
                    source.copyTo(target, overwrite = false)
                }
            }
        }
        val oldSupertonic = File(files, "models/supertonic3")
        val newSupertonic = supertonicDir(context)
        if (oldSupertonic.isDirectory) {
            check(newSupertonic.mkdirs() || newSupertonic.isDirectory) {
                "Nie można utworzyć katalogu modelu Supertonic."
            }
            oldSupertonic.walkTopDown().filter { it.isFile }.forEach { source ->
                val target = File(newSupertonic, source.relativeTo(oldSupertonic).path)
                if (!target.exists()) {
                    target.parentFile?.mkdirs()
                    source.copyTo(target, overwrite = false)
                }
            }
        }

        val oldVoices = File(files, "voice_profiles")
        val newVoices = voicesDir(context)
        if (oldVoices.isDirectory) {
            check(newVoices.mkdirs() || newVoices.isDirectory) { "Nie można utworzyć katalogu profili." }
            oldVoices.walkTopDown().filter { it.isFile }.forEach { source ->
                val relative = source.relativeTo(oldVoices)
                val target = File(newVoices, relative.path)
                if (!target.exists()) {
                    target.parentFile?.mkdirs()
                    source.copyTo(target, overwrite = false)
                }
            }
        }
        val oldOutput = File(files, "audio")
        val newOutput = outputDir(context)
        if (oldOutput.isDirectory) {
            check(newOutput.mkdirs() || newOutput.isDirectory) { "Nie można utworzyć katalogu wyników." }
            oldOutput.walkTopDown().filter { it.isFile }.forEach { source ->
                val target = File(newOutput, source.relativeTo(oldOutput).path)
                if (!target.exists()) {
                    target.parentFile?.mkdirs()
                    source.copyTo(target, overwrite = false)
                }
            }
        }
    }
}
