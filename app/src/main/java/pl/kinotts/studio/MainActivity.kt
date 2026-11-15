package pl.kinotts.studio

import android.app.ActivityManager
import android.os.Debug
import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.kinotts.studio.analysis.AudioAnalysisResult
import pl.kinotts.studio.analysis.AudioAnalyzer
import pl.kinotts.studio.engine.SupertonicEngine
import pl.kinotts.studio.engine.XtTsOnnxRuntime
import pl.kinotts.studio.models.ModelManager
import pl.kinotts.studio.models.XttsModelManager
import pl.kinotts.studio.security.HfTokenStore
import pl.kinotts.studio.voices.VoiceProfile
import pl.kinotts.studio.voices.VoiceProfileStore
import pl.kinotts.studio.xtts.XttsSynthesizer
import pl.kinotts.studio.xtts.XttsSamplingSettings
import pl.kinotts.studio.storage.KinoStorage
import java.io.File
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { KinoTtsApp() }
    }

    override fun onResume() {
        super.onResume()
        if (KinoStorage.hasAccess(this)) {
            runCatching { KinoStorage.ensureDirectories(this) }
        }
    }
}

private enum class Screen(val title: String) {
    Studio("Studio"), Voices("Głosy"), Models("Modele"), Settings("Ustawienia")
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun KinoTtsApp() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val models = remember { ModelManager(context.applicationContext) }
    val xttsModels = remember { XttsModelManager(context.applicationContext) }
    val xttsPreferences = remember {
        context.getSharedPreferences("kinotts_settings", android.content.Context.MODE_PRIVATE)
    }
    var xttsExecutionProvider by remember {
        mutableStateOf(if (xttsPreferences.getString("xtts_execution_provider", "cpu") == "nnapi") "nnapi" else "cpu")
    }
    val analyzer = remember { AudioAnalyzer(context.applicationContext) }
    val profileStore = remember { VoiceProfileStore(context.applicationContext) }
    val supertonic = remember { SupertonicEngine(KinoStorage.outputDir(context).apply { mkdirs() }) }
    val xttsRuntime = remember { XttsSynthesizer(xttsModels.modelDir(), KinoStorage.outputDir(context).apply { mkdirs() }, initialExecutionProvider = xttsExecutionProvider) }

    // KinoTTS_ADVANCED_SETTINGS_PATCH_V2
    val initialAppLanguage = remember {
        val saved = xttsPreferences.getString("app_language", null)
        val detected = saved?.takeIf { it == "pl" || it == "en" }
            ?: if (Locale.getDefault().language.equals("pl", ignoreCase = true)) "pl" else "en"
        if (saved == null) xttsPreferences.edit().putString("app_language", detected).apply()
        detected
    }
    var appLanguage by remember { mutableStateOf(initialAppLanguage) }
    var autoSyncLanguage by remember { mutableStateOf(xttsPreferences.getBoolean("auto_sync_language", true)) }
    var xttsLanguage by remember { mutableStateOf(
        if (autoSyncLanguage) initialAppLanguage else (xttsPreferences.getString("xtts_language", initialAppLanguage) ?: initialAppLanguage)
    ) }
    var supertonicLanguage by remember { mutableStateOf(
        if (autoSyncLanguage) initialAppLanguage else (xttsPreferences.getString("supertonic_language", initialAppLanguage) ?: initialAppLanguage)
    ) }
    var xttsSpeed by remember { mutableFloatStateOf(xttsPreferences.getFloat("xtts_speed", 1f).coerceIn(0.5f, 2f)) }
    var supertonicSpeed by remember { mutableFloatStateOf(xttsPreferences.getFloat("supertonic_speed", 1f).coerceIn(0.6f, 1.6f)) }
    var supertonicThreads by remember { mutableIntStateOf(xttsPreferences.getInt("supertonic_threads", 2).coerceIn(1, 8)) }
    var sid by remember { mutableIntStateOf(xttsPreferences.getInt("supertonic_sid", 0).coerceIn(0, 9)) }
    var steps by remember { mutableIntStateOf(xttsPreferences.getInt("supertonic_steps", 8).coerceIn(4, 128)) }
    var xttsTemperature by remember { mutableFloatStateOf(xttsPreferences.getFloat("xtts_temperature", 0.75f).coerceIn(0.1f, 2f)) }
    var xttsTopK by remember { mutableIntStateOf(xttsPreferences.getInt("xtts_top_k", 50).coerceIn(1, 100)) }
    var xttsTopP by remember { mutableFloatStateOf(xttsPreferences.getFloat("xtts_top_p", 0.85f).coerceIn(0.1f, 1f)) }
    var xttsRepetitionPenalty by remember { mutableFloatStateOf(xttsPreferences.getFloat("xtts_repetition_penalty", 1.1f).coerceIn(1f, 2f)) }

    var screen by remember { mutableStateOf(Screen.Studio) }
    var engine by remember { mutableStateOf("Supertonic 3") }
    var text by remember { mutableStateOf(defaultPrompt(initialAppLanguage)) }
    var selectedAudio by remember { mutableStateOf<Uri?>(null) }
    var selectedAudioName by remember { mutableStateOf("Nie wybrano pliku") }
    var analysis by remember { mutableStateOf<AudioAnalysisResult?>(null) }
    var profileName by remember { mutableStateOf("Mój głos") }
    var consentConfirmed by remember { mutableStateOf(false) }
    var profiles by remember { mutableStateOf(profileStore.list()) }
    val voiceGroups = profiles.groupBy { it.groupId }.values.toList()
    var xttsChosenProfileId by remember { mutableStateOf(profiles.firstOrNull()?.groupId.orEmpty()) }
    var xttsSynthesizing by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("Gotowy") }
    var statusVisible by remember { mutableStateOf(false) }
    var pendingModelDelete by remember { mutableStateOf<String?>(null) }
    var appendToProfileId by remember { mutableStateOf("") }
    var computeBackendStatus by remember { mutableStateOf("Brak aktywnej inferencji — oczekiwanie na generowanie") }
    var hfToken by remember { mutableStateOf(HfTokenStore.load(context).orEmpty()) }
    var xttsInstalled by remember { mutableStateOf(xttsModels.isInstalled()) }
    var xttsModelStatus by remember {
        mutableStateOf(if (xttsInstalled) "Model gotowy" else "Model do pobrania")
    }
    var supertonicModelStatus by remember {
        mutableStateOf(if (models.isSupertonicInstalled()) "Model gotowy" else "Model do pobrania")
    }
    var xttsGraphInfo by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
    var progressEta by remember { mutableStateOf("") }
    var ramStats by remember { mutableStateOf("Odczyt pamięci…") }
    var ramWarning by remember { mutableStateOf(false) }
    var ortStatus by remember { mutableStateOf("Sprawdzanie ONNX Runtime…") }
    var output by remember { mutableStateOf<File?>(null) }
    var player by remember { mutableStateOf<MediaPlayer?>(null) }

    fun tr(polish: String, english: String): String = if (appLanguage == "pl") polish else english

    fun selectAppLanguage(next: String) {
        if (next != "pl" && next != "en") return
        val previous = appLanguage
        if (previous == next) return
        appLanguage = next
        xttsPreferences.edit().putString("app_language", next).apply()
        if (text == defaultPrompt(previous)) text = defaultPrompt(next)
        if (autoSyncLanguage) {
            xttsLanguage = next
            supertonicLanguage = next
            xttsPreferences.edit()
                .putString("xtts_language", next)
                .putString("supertonic_language", next)
                .apply()
        }
    }

    fun setAutoSyncLanguage(enabled: Boolean) {
        autoSyncLanguage = enabled
        val editor = xttsPreferences.edit().putBoolean("auto_sync_language", enabled)
        if (enabled) {
            xttsLanguage = appLanguage
            supertonicLanguage = appLanguage
            editor.putString("xtts_language", appLanguage)
                .putString("supertonic_language", appLanguage)
        }
        editor.apply()
    }

    fun selectXttsLanguage(next: String) {
        xttsLanguage = next
        autoSyncLanguage = false
        xttsPreferences.edit().putBoolean("auto_sync_language", false).putString("xtts_language", next).apply()
    }

    fun selectSupertonicLanguage(next: String) {
        supertonicLanguage = next
        autoSyncLanguage = false
        xttsPreferences.edit().putBoolean("auto_sync_language", false).putString("supertonic_language", next).apply()
    }

    val pickAudio = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            try {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: Exception) { }
            selectedAudio = uri
            analysis = null
            consentConfirmed = false
            selectedAudioName = try {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                    ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                    ?: uri.lastPathSegment
                    ?: "Wybrany plik audio/wideo"
            } catch (_: Exception) {
                uri.lastPathSegment ?: "Wybrany plik audio/wideo"
            }
            status = "Plik wybrany. Możesz wykonać lokalną analizę audio."
        }
    }

    val exportAudio = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        val source = output
        if (uri != null && source != null) {
            scope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        val target = context.contentResolver.openOutputStream(uri)
                            ?: error("Nie można otworzyć wybranego miejsca zapisu")
                        target.use { out -> source.inputStream().use { input -> input.copyTo(out) } }
                    }
                    status = "Zapisano plik WAV."
                } catch (e: Exception) {
                    status = "Błąd zapisu: ${e.message ?: e::class.java.simpleName}"
                }
            }
        }
    }

    LaunchedEffect(status, busy) {
        statusVisible = status.isNotBlank() && status != "Gotowy"
        if (!busy && statusVisible) {
            delay(5000)
            statusVisible = false
        }
    }

    LaunchedEffect(Unit) {
        val manager = context.getSystemService(android.content.Context.ACTIVITY_SERVICE) as ActivityManager
        val memory = ActivityManager.MemoryInfo()
        while (true) {
            manager.getMemoryInfo(memory)
            val appMb = Debug.getPss() / 1024.0
            val availableMb = memory.availMem / (1024.0 * 1024.0)
            val totalMb = memory.totalMem / (1024.0 * 1024.0)
            ramStats = String.format(Locale.US,
                "Aplikacja: %.0f MB • wolna RAM: %.0f / %.0f MB",
                appMb, availableMb, totalMb)
            ramWarning = memory.lowMemory || availableMb < 500.0
            delay(2500)
        }
    }

    LaunchedEffect(engine) {
        ortStatus = if (engine == "XTTS-v2") {
            withContext(Dispatchers.IO) { XtTsOnnxRuntime.diagnose().message }
        } else {
            "Niezaładowany — aktywny jest Supertonic 3."
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            player?.release()
            supertonic.close()
            xttsRuntime.close()
        }
    }

    MaterialTheme(colorScheme = darkColorScheme()) {
        pendingModelDelete?.let { targetModel ->
            AlertDialog(
                onDismissRequest = { pendingModelDelete = null },
                title = { Text(tr("Usunąć model?", "Delete model?")) },
                text = {
                    Text(
                        if (targetModel == "xtts")
                            "Model XTTS-v2 zostanie usunięty z Documents/KinoTTS/Models. Profile głosu i wygenerowane nagrania pozostaną."
                        else
                            "Model Supertonic 3 zostanie usunięty z Documents/KinoTTS/Models. Wygenerowane nagrania pozostaną."
                    )
                },
                confirmButton = {
                    TextButton(onClick = {
                        pendingModelDelete = null
                        scope.launch {
                            busy = true
                            progress = 0f
                            status = if (targetModel == "xtts") "Usuwam model XTTS-v2…" else "Usuwam model Supertonic 3…"
                            try {
                                withContext(Dispatchers.IO) {
                                    if (targetModel == "xtts") {
                                        xttsRuntime.unload()
                                        check(xttsModels.deleteInstalled()) { "Nie udało się całkowicie usunąć modelu XTTS." }
                                    } else {
                                        supertonic.close()
                                        check(models.deleteSupertonic()) { "Nie udało się całkowicie usunąć modelu Supertonic." }
                                    }
                                }
                                if (targetModel == "xtts") {
                                    xttsInstalled = false
                                    xttsGraphInfo = ""
                                    xttsModelStatus = "Model do pobrania"
                                    status = "Usunięto model XTTS-v2."
                                } else {
                                    supertonicModelStatus = "Model do pobrania"
                                    status = "Usunięto model Supertonic 3."
                                }
                            } catch (e: Exception) {
                                status = "Błąd usuwania modelu: ${e.message ?: e::class.java.simpleName}"
                            } finally {
                                busy = false
                            }
                        }
                    }) { Text(tr("Usuń", "Delete")) }
                },
                dismissButton = {
                    TextButton(onClick = { pendingModelDelete = null }) { Text(tr("Anuluj", "Cancel")) }
                }
            )
        }
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Column {
                            Text("KinoTTS Studio", fontWeight = FontWeight.Bold)
                            Text(
                                "Lokalne modele • przetwarzanie na urządzeniu",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                )
            },
            bottomBar = {
                Column {
                    Card(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Column(
                            Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(5.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("XTTS-v2", modifier = Modifier.width(88.dp), fontWeight = FontWeight.Bold, maxLines = 1)
                                Text(
                                    xttsModelStatus,
                                    modifier = Modifier.weight(1f),
                                    style = MaterialTheme.typography.bodySmall,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("Supertonic 3", modifier = Modifier.width(88.dp), fontWeight = FontWeight.Bold, maxLines = 1)
                                Text(
                                    supertonicModelStatus,
                                    modifier = Modifier.weight(1f),
                                    style = MaterialTheme.typography.bodySmall,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(tr("Obliczenia", "Compute"), modifier = Modifier.width(88.dp), fontWeight = FontWeight.Bold, maxLines = 1)
                                Text(computeBackendStatus, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall,
                                    maxLines = 2, overflow = TextOverflow.Ellipsis)
                            }
                            if (busy) {
                                LinearProgressIndicator(
                                    progress = { progress.coerceIn(0f, 1f) },
                                    modifier = Modifier.fillMaxWidth()
                                )
                                if (progressEta.isNotBlank()) {
                                    Text(progressEta, style = MaterialTheme.typography.labelSmall)
                                }
                            }
                        }
                    }
                    if (statusVisible || busy) StatusCard(if (appLanguage == "en") localizeStatusEnglish(status) else status)
                    NavigationBar {
                        Screen.entries.forEach { item ->
                            val icon = when (item) {
                                Screen.Studio -> Icons.Default.Home
                                Screen.Voices -> Icons.Default.GraphicEq
                                Screen.Models -> Icons.Default.Download
                                Screen.Settings -> Icons.Default.Settings
                            }
                            NavigationBarItem(
                                selected = screen == item,
                                onClick = { screen = item },
                                icon = { Icon(icon, contentDescription = screenTitle(item, appLanguage == "en")) },
                                label = { Text(screenTitle(item, appLanguage == "en")) }
                            )
                        }
                    }
                }
            }
        ) { padding ->
            Column(
                modifier = Modifier.fillMaxSize().padding(padding)
                    .verticalScroll(rememberScrollState()).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                when (screen) {
                    Screen.Studio -> {
                        Text(tr("Generator mowy", "Speech generator"), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                Text(tr("Silnik", "Engine"), style = MaterialTheme.typography.titleMedium)
                                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                                    listOf("XTTS-v2", "Supertonic 3").forEachIndexed { index, name ->
                                        SegmentedButton(
                                            selected = engine == name,
                                            onClick = {
                                                if (!busy && engine != name) {
                                                    scope.launch {
                                                        busy = true
                                                        status = "Przełączam silnik na $name…"
                                                        try {
                                                            withContext(Dispatchers.IO) {
                                                                if (name == "XTTS-v2") supertonic.close()
                                                                else xttsRuntime.unload()
                                                            }
                                                            engine = name
                                                            status = "Przełączono silnik: $name"
                                                        } catch (e: Exception) {
                                                            status = "Nie udało się przełączyć silnika: ${e.message ?: e::class.java.simpleName}"
                                                        } finally {
                                                            busy = false
                                                        }
                                                    }
                                                }
                                            },
                                            shape = SegmentedButtonDefaults.itemShape(index, 2)
                                        ) { Text(name) }
                                    }
                                }

                                if (engine == "XTTS-v2") {
                                    Text(
                                        "XTTS-v2 działa lokalnie przez ONNX Runtime. Dodaj własną próbkę albo nagranie użyte za zgodą właściciela, zapisz profil i generuj mowę offline.",
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    OutlinedButton(
                                        onClick = { pickAudio.launch(arrayOf("audio/*", "video/mp4")) },
                                        modifier = Modifier.fillMaxWidth(), enabled = !busy
                                    ) {
                                        Icon(Icons.Default.AudioFile, contentDescription = null)
                                        Spacer(Modifier.width(8.dp)); Text(tr("Wybierz MP3 / MP4 / WAV", "Choose MP3 / MP4 / WAV"))
                                    }
                                    Text(selectedAudioName, style = MaterialTheme.typography.bodySmall)
                                    Button(
                                        onClick = {
                                            val uri = selectedAudio
                                            if (uri != null) scope.launch {
                                                busy = true; progress = 0f; status = "Dekoduję próbkę audio…"
                                                try {
                                                    analysis = withContext(Dispatchers.IO) {
                                                        analyzer.analyze(uri) { value -> if (ProgressLimiter.allow("analysis", 90L) || value <= 0f || value >= 1f) {
                                                            scope.launch(Dispatchers.Main.immediate) { progress = value.coerceIn(0f, 1f) }
                                                        } }
                                                    }
                                                    status = "Próbka gotowa. Zapisz ją jako profil głosu."
                                                } catch (e: Exception) {
                                                    analysis = null
                                                    status = "Błąd analizy: ${e.message ?: e::class.java.simpleName}"
                                                } finally { busy = false }
                                            }
                                        }, enabled = selectedAudio != null && !busy, modifier = Modifier.fillMaxWidth()
                                    ) { Text(if (busy) "Analizuję…" else "Analizuj próbkę") }
                                    analysis?.let { result ->
                                        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                                            Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                                                Text(tr("Parametry próbki", "Sample settings"), style = MaterialTheme.typography.titleMedium)
                                                Text(tr("Czas: ${formatSeconds(result.durationSeconds)} • ${result.sampleRate} Hz • ${result.sourceChannels} kan.", "Duration: ${formatSeconds(result.durationSeconds)} • ${result.sampleRate} Hz • ${result.sourceChannels} channels"))
                                                Text(tr("RMS: ${formatNumber(result.rmsDbfs)} dBFS • clipping: ${formatNumber(result.clippingPercent)}% • cisza: ${formatNumber(result.silencePercent)}%", "RMS: ${formatNumber(result.rmsDbfs)} dBFS • clipping: ${formatNumber(result.clippingPercent)}% • silence: ${formatNumber(result.silencePercent)}%"))
                                            }
                                        }
                                        var appendGroupExpanded by remember { mutableStateOf(false) }
                                        val appendGroup = voiceGroups.firstOrNull { it.first().groupId == appendToProfileId }
                                        ExposedDropdownMenuBox(expanded = appendGroupExpanded,
                                            onExpandedChange = { appendGroupExpanded = it }) {
                                            OutlinedTextField(
                                                value = appendGroup?.first()?.name ?: "Utwórz nowy głos",
                                                onValueChange = {}, readOnly = true,
                                                modifier = Modifier.menuAnchor().fillMaxWidth(),
                                                label = { Text(tr("Przypisz próbkę do głosu", "Assign sample to a voice")) }
                                            )
                                            ExposedDropdownMenu(expanded = appendGroupExpanded,
                                                onDismissRequest = { appendGroupExpanded = false }) {
                                                DropdownMenuItem(text = { Text(tr("+ Utwórz nowy głos", "+ Create a new voice")) }, onClick = {
                                                    appendToProfileId = ""; appendGroupExpanded = false
                                                })
                                                voiceGroups.forEach { group ->
                                                    DropdownMenuItem(
                                                        text = { Text(tr("${group.first().name} (${group.size} próbek)", "${group.first().name} (${group.size} samples)")) },
                                                        onClick = {
                                                            appendToProfileId = group.first().groupId
                                                            profileName = group.first().name
                                                            appendGroupExpanded = false
                                                        }
                                                    )
                                                }
                                            }
                                        }
                                        OutlinedTextField(
                                            value = profileName, onValueChange = { profileName = it },
                                            label = { Text(tr("Nazwa nowego głosu", "New voice name")) }, modifier = Modifier.fillMaxWidth(),
                                            singleLine = true, enabled = appendGroup == null
                                        )
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Checkbox(checked = consentConfirmed, onCheckedChange = { consentConfirmed = it })
                                            Text(tr("To mój głos albo mam wyraźną zgodę właściciela nagrania.", "This is my voice, or I have the recording owner’s explicit permission."), style = MaterialTheme.typography.bodySmall)
                                        }
                                        Button(
                                            onClick = {
                                                scope.launch {
                                                    busy = true
                                                    try {
                                                        val targetGroup = appendToProfileId.takeIf { id ->
                                                            voiceGroups.any { it.first().groupId == id }
                                                        }
                                                        val saved = withContext(Dispatchers.IO) {
                                                            profileStore.save(profileName, result, xttsLanguage, consentConfirmed, targetGroup)
                                                        }
                                                        profiles = withContext(Dispatchers.IO) { profileStore.list() }
                                                        xttsChosenProfileId = saved.groupId
                                                        consentConfirmed = false
                                                        status = if (targetGroup == null)
                                                            "Utworzono głos ${saved.name} i dodano pierwszą próbkę."
                                                        else
                                                            "Dodano próbkę do głosu ${saved.name}. Łącznie: ${profiles.count { it.groupId == saved.groupId }}."
                                                    } catch (e: Exception) {
                                                        status = "Błąd zapisu profilu: ${e.message ?: e::class.java.simpleName}"
                                                    } finally { busy = false }
                                                }
                                            }, enabled = !busy && consentConfirmed && profileName.trim().length in 2..40,
                                            modifier = Modifier.fillMaxWidth()
                                        ) { Text(tr("Zapisz profil głosu", "Save voice profile")) }
                                    }

                                    Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                    Text(tr("Tekst do wygenerowania", "Text to synthesize"), style = MaterialTheme.typography.titleMedium)
                                    OutlinedTextField(
                                        value = text, onValueChange = { text = it }, label = { Text(tr("Tekst po polsku lub w innym języku XTTS", "Text in Polish or another XTTS language")) },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp), minLines = 3
                                    )
                                    var xttsLanguageExpanded by remember { mutableStateOf(false) }
                                    ExposedDropdownMenuBox(expanded = xttsLanguageExpanded, onExpandedChange = { xttsLanguageExpanded = it }) {
                                        OutlinedTextField(
                                            value = xttsLanguage, onValueChange = {}, readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text(tr("Język XTTS", "XTTS language")) }
                                        )
                                        ExposedDropdownMenu(expanded = xttsLanguageExpanded, onDismissRequest = { xttsLanguageExpanded = false }) {
                                            listOf("pl", "en", "de", "fr", "es", "it", "pt", "cs", "ru", "nl", "tr", "zh", "ja", "hu", "ko", "ar", "hi").forEach { lang ->
                                                DropdownMenuItem(text = { Text(lang) }, onClick = { selectXttsLanguage(lang); xttsLanguageExpanded = false })
                                            }
                                        }
                                    }
                                    val selectedXttsProfile = voiceGroups.firstOrNull { it.first().groupId == xttsChosenProfileId }?.first()
                                        ?: voiceGroups.firstOrNull()?.first()
                                    var xttsProfileExpanded by remember { mutableStateOf(false) }
                                    ExposedDropdownMenuBox(expanded = xttsProfileExpanded, onExpandedChange = { xttsProfileExpanded = it }) {
                                        OutlinedTextField(
                                            value = selectedXttsProfile?.let { selected ->
                                                val count = voiceGroups.firstOrNull { it.first().groupId == selected.groupId }?.size ?: 1
                                                "${selected.name} ($count próbek)"
                                            } ?: "Brak zapisanego profilu",
                                            onValueChange = {}, readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text(tr("Głos do sklonowania", "Voice to clone")) }
                                        )
                                        ExposedDropdownMenu(expanded = xttsProfileExpanded, onDismissRequest = { xttsProfileExpanded = false }) {
                                            voiceGroups.forEach { group ->
                                                DropdownMenuItem(
                                                    text = { Text(tr("${group.first().name} (${group.size} próbek)", "${group.first().name} (${group.size} samples)")) },
                                                    onClick = {
                                                        xttsChosenProfileId = group.first().groupId
                                                        xttsProfileExpanded = false
                                                    }
                                                )
                                            }
                                        }
                                    }
                                    Text(tr("Tempo: ${formatNumber(xttsSpeed)}×", "Speed: ${formatNumber(xttsSpeed)}×"))
                                    Slider(value = xttsSpeed, onValueChange = { xttsSpeed = it; xttsPreferences.edit().putFloat("xtts_speed", it).apply() }, valueRange = 0.5f..2.0f)
                                    if (!xttsInstalled) {
                                        Text(tr("Najpierw pobierz XTTS-v2 w zakładce Modele.", "Download XTTS-v2 first in the Models tab."), color = MaterialTheme.colorScheme.error)
                                    }
                                    if (profiles.isEmpty()) {
                                        Text(tr("Najpierw zaimportuj próbkę i zapisz profil głosu.", "Import a sample and save a voice profile first."), color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                    Button(
                                        onClick = {
                                            val selected = selectedXttsProfile
                                            if (selected != null) scope.launch {
                                                busy = true; xttsSynthesizing = true; progress = 0f; status = "Uruchamiam XTTS-v2…"
                                                try {
                                                    computeBackendStatus = "XTTS: inicjalizacja backendu ${xttsExecutionProvider.uppercase(Locale.ROOT)}…"
                                                    withContext(Dispatchers.IO) { xttsRuntime.open() }
                                                    computeBackendStatus = xttsRuntime.executionProviderStatus
                                                    val referenceWav = withContext(Dispatchers.IO) {
                                                        profileStore.combinedReference(selected.groupId)
                                                    }
                                                    val wav = withContext(Dispatchers.IO) {
                                                        xttsRuntime.synthesize(text, xttsLanguage, referenceWav, xttsSpeed, sampling = XttsSamplingSettings(xttsTemperature, xttsTopK, xttsTopP, xttsRepetitionPenalty)) { value ->
                                                            if (ProgressLimiter.allow("xtts", 90L) || value <= 0f || value >= 1f) {
                                                                scope.launch(Dispatchers.Main.immediate) { progress = value.coerceIn(0f, 1f) }
                                                            }
                                                        }
                                                    }
                                                    output = wav
                                                    status = "XTTS wygenerował WAV: ${wav.name}"
                                                } catch (e: Exception) {
                                                    status = "Błąd XTTS: ${e.message ?: e::class.java.simpleName}"
                                                } finally { xttsSynthesizing = false; busy = false }
                                            }
                                        },
                                        enabled = !busy && xttsInstalled && selectedXttsProfile != null && text.isNotBlank(),
                                        modifier = Modifier.fillMaxWidth()
                                    ) { Text(if (busy) tr("Generuję głos…", "Synthesizing…") else tr("Generuj mowę XTTS", "Synthesize with XTTS")) }
                                    if (xttsSynthesizing) {
                                        OutlinedButton(
                                            onClick = { xttsRuntime.cancel(); status = "Wysłano żądanie anulowania XTTS…" },
                                            modifier = Modifier.fillMaxWidth()
                                        ) { Text(tr("Anuluj generowanie", "Cancel synthesis")) }
                                    }
                                } else {
                                    OutlinedTextField(
                                        value = text,
                                        onValueChange = { text = it },
                                        label = { Text(tr("Tekst do przeczytania", "Text to read")) },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 140.dp),
                                        minLines = 4
                                    )
                                    Text(tr("Głos: ${sid + 1} / 10", "Voice: ${sid + 1} / 10"))
                                    Slider(value = sid.toFloat(), onValueChange = { sid = it.toInt(); xttsPreferences.edit().putInt("supertonic_sid", sid).apply() }, valueRange = 0f..9f, steps = 8)
                                    Text(tr("Język", "Language"))
                                    var languageExpanded by remember { mutableStateOf(false) }
                                    ExposedDropdownMenuBox(expanded = languageExpanded, onExpandedChange = { languageExpanded = it }) {
                                        OutlinedTextField(
                                            value = supertonicLanguage,
                                            onValueChange = {},
                                            readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(),
                                            label = { Text(tr("Kod języka", "Language code")) }
                                        )
                                        ExposedDropdownMenu(expanded = languageExpanded, onDismissRequest = { languageExpanded = false }) {
                                            SupertonicEngine.SUPPORTED_LANGUAGES.sorted().forEach { lang ->
                                                DropdownMenuItem(
                                                    text = { Text(lang) },
                                                    onClick = { selectSupertonicLanguage(lang); languageExpanded = false }
                                                )
                                            }
                                        }
                                    }
                                    Text(tr("Tempo: ${formatNumber(supertonicSpeed)}×", "Speed: ${formatNumber(supertonicSpeed)}×"))
                                    Slider(value = supertonicSpeed, onValueChange = { supertonicSpeed = it; xttsPreferences.edit().putFloat("supertonic_speed", it).apply() }, valueRange = 0.6f..1.6f)
                                    Text(tr("Kroki: $steps", "Steps: $steps"))
                                    Slider(value = steps.toFloat(),
                                        onValueChange = { steps = it.toInt(); xttsPreferences.edit().putInt("supertonic_steps", steps).apply() },
                                        valueRange = 4f..128f, steps = 123)
                                }
                            }
                        }

                        if (engine == "Supertonic 3") {
                            Button(
                                onClick = {
                                    scope.launch {
                                        busy = true
                                        progress = 0f
                                        status = "Sprawdzam model Supertonic 3…"
                                        try {
                                            val modelDir = withContext(Dispatchers.IO) {
                                                models.ensureSupertonic { value -> if (ProgressLimiter.allow("supertonic-synth", 100L) || value <= 0f || value >= 1f) {
                                                    scope.launch(Dispatchers.Main.immediate) { progress = value.coerceIn(0f, 1f) }
                                                } }
                                            }
                                            status = "Ładuję model i generuję mowę…"
                                            withContext(Dispatchers.IO) { supertonic.load(modelDir, supertonicThreads) }
                                            computeBackendStatus = supertonic.executionProviderStatus
                                            val wav = withContext(Dispatchers.IO) {
                                                supertonic.generate(text, sid, supertonicSpeed, steps, supertonicLanguage)
                                            }
                                            output = wav
                                            status = "Gotowe: ${wav.name}"
                                        } catch (e: LinkageError) {
                                            status = "Błąd biblioteki natywnej Supertonic: ${e.message ?: e.javaClass.simpleName}"
                                        } catch (e: OutOfMemoryError) {
                                            status = "Za mało pamięci dla syntezy Supertonic. Skróć tekst i zamknij aplikacje działające w tle."
                                        } catch (e: Exception) {
                                            status = "Błąd syntezy: ${e.message ?: e::class.java.simpleName}"
                                        } finally {
                                            busy = false
                                        }
                                    }
                                },
                                enabled = !busy && text.isNotBlank(),
                                modifier = Modifier.fillMaxWidth()
                            ) { Text(if (busy) tr("Pracuję…", "Working…") else tr("Generuj mowę", "Synthesize speech")) }
                        }

                        if (busy) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                        output?.let { file ->
                            Card {
                                Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text(tr("Wynik audio", "Audio output"), style = MaterialTheme.typography.titleMedium)
                                    Text(file.name, style = MaterialTheme.typography.bodySmall)
                                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        OutlinedButton(onClick = {
                                            player?.release()
                                            player = MediaPlayer().apply {
                                                setDataSource(file.absolutePath)
                                                prepare()
                                                start()
                                            }
                                        }) { Text(tr("Odtwórz", "Play")) }
                                        OutlinedButton(onClick = { exportAudio.launch(file.name) }) {
                                            Text(tr("Zapisz WAV", "Save WAV"))
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Screen.Voices -> {
                        Text(tr("Biblioteka głosów", "Voice library"), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Supertonic 3", style = MaterialTheme.typography.titleMedium)
                                Text(tr("10 gotowych stylów głosu (ID 0–9). Wybierz głos w Studio.", "10 built-in voice styles (IDs 0–9). Choose a voice in Studio."))
                            }
                        }
                        Text(tr("Zapisane próbki", "Saved samples"), style = MaterialTheme.typography.titleMedium)
                        if (voiceGroups.isEmpty()) {
                            Text(tr("Nie zapisano jeszcze żadnego głosu.", "No voice samples have been saved yet."), style = MaterialTheme.typography.bodySmall)
                        } else {
                            voiceGroups.forEach { group ->
                                Card {
                                    Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Text(group.first().name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                                        Text(if (appLanguage == "pl") "${group.size} próbek przypisanych do jednego głosu. XTTS łączy je w referencję do syntezy." else "${group.size} samples grouped under one voice. XTTS combines them into a synthesis reference.",
                                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        group.forEachIndexed { sampleIndex, sample ->
                                            VoiceProfileCard(sample.copy(name = if (appLanguage == "pl") "Próbka ${sampleIndex + 1}" else "Sample ${sampleIndex + 1}"), english = appLanguage == "en", onDelete = {
                                                if (!busy) scope.launch {
                                                    try {
                                                        val deleted = withContext(Dispatchers.IO) { profileStore.delete(sample.id) }
                                                        profiles = withContext(Dispatchers.IO) { profileStore.list() }
                                                        if (xttsChosenProfileId == sample.groupId && profiles.none { it.groupId == sample.groupId }) {
                                                            xttsChosenProfileId = profiles.firstOrNull()?.groupId.orEmpty()
                                                        }
                                                        status = if (deleted) "Usunięto próbkę z głosu ${sample.name}" else "Nie udało się usunąć pliku próbki"
                                                    } catch (e: Exception) {
                                                        status = "Nie udało się usunąć próbki: ${e.message ?: e::class.java.simpleName}"
                                                    }
                                                }
                                            }, enabled = !busy)
                                        }
                                    }
                                }
                            }
                        }
                        Text(
                            "Zapisane próbki mogą służyć jako profile referencyjne dla lokalnego silnika XTTS-v2.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Screen.Models -> {
                        Text(tr("Menedżer modeli", "Model manager"), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text("Supertonic 3 INT8", style = MaterialTheme.typography.titleMedium)
                                Text(if (models.isSupertonicInstalled()) "Pobrany i zweryfikowany w Documents/KinoTTS/Models" else "Niepobrany — pobierze się przy pierwszej syntezie")
                                Button(
                                    onClick = {
                                        scope.launch {
                                            busy = true
                                            progress = 0f
                                            progressEta = "Szacowanie czasu…"
                                            status = "Pobieram i weryfikuję Supertonic 3…"
                                            supertonicModelStatus = "Pobieranie modelu…"
                                            val startedAt = System.currentTimeMillis()
                                            try {
                                                withContext(Dispatchers.IO) {
                                                    models.ensureSupertonic { value ->
                                                        if (ProgressLimiter.allow("supertonic-download", 120L) || value <= 0f || value >= 1f) {
                                                            scope.launch(Dispatchers.Main.immediate) {
                                                                progress = value.coerceIn(0f, 1f)
                                                                progressEta = estimateRemainingLabel(value, startedAt)
                                                            }
                                                        }
                                                    }
                                                }
                                                supertonicModelStatus = "Model gotowy"
                                                status = "Model Supertonic 3 jest gotowy"
                                            } catch (e: Exception) {
                                                supertonicModelStatus = "Błąd: ${e.message ?: e::class.java.simpleName}"
                                                status = "Błąd pobierania: ${e.message ?: e::class.java.simpleName}"
                                            } finally {
                                                busy = false
                                            }
                                        }
                                    },
                                    enabled = !busy,
                                    modifier = Modifier.fillMaxWidth()
                                ) { Text(tr("Pobierz / sprawdź model", "Download / verify model")) }
                                if (models.isSupertonicInstalled()) {
                                    OutlinedButton(onClick = { pendingModelDelete = "supertonic" }, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                                        Text(tr("Usuń model Supertonic 3", "Delete Supertonic 3 model"))
                                    }
                                }
                            }
                        }
                        Card {
                            Column(
                                Modifier.padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {

                                Text(
                                    "XTTS-v2 ONNX • GPT INT8",
                                    style = MaterialTheme.typography.titleMedium
                                )

                                Text(
                                    if (xttsInstalled)
                                        "Model gotowy"
                                    else
                                        "Pobierz model z Hugging Face. Pliki pozostaną na telefonie."
                                )

                                OutlinedTextField(
                                    value = hfToken,
                                    onValueChange = { hfToken = it },
                                    label = { Text(tr("Hugging Face token (hf_… )", "Hugging Face token (hf_… )")) },
                                    supportingText = {
                                        Text(
                                            "Token jest szyfrowany kluczem Android Keystore i zapamiętywany na tym urządzeniu."
                                        )
                                    },
                                    visualTransformation = PasswordVisualTransformation(),
                                    singleLine = true,
                                    enabled = !busy,
                                    modifier = Modifier.fillMaxWidth()
                                )

                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    OutlinedButton(onClick = {
                                        try {
                                            HfTokenStore.save(context, hfToken)
                                            status = "Token Hugging Face zapisany bezpiecznie na tym urządzeniu."
                                        } catch (e: Exception) {
                                            status = "Nie udało się zapisać tokena: ${e.message ?: e::class.java.simpleName}"
                                        }
                                    }, enabled = !busy && hfToken.trim().startsWith("hf_"), modifier = Modifier.weight(1f)) {
                                        Text(tr("Zapisz token", "Save token"))
                                    }
                                    OutlinedButton(onClick = {
                                        HfTokenStore.clear(context)
                                        hfToken = ""
                                        status = "Usunięto zapisany token Hugging Face."
                                    }, enabled = !busy, modifier = Modifier.weight(1f)) {
                                        Text(tr("Usuń token", "Delete token"))
                                    }
                                }

                                Button(
                                    onClick = {

                                        val tokenForDownload = hfToken.trim()

                                        scope.launch {

                                            busy = true
                                            progress = 0f
                                            xttsGraphInfo = ""

                                            xttsModelStatus =
                                                "Pobieram i weryfikuję pliki XTTS-v2…"
                                            progressEta = "Szacowanie czasu…"
                                            val startedAt = System.currentTimeMillis()

                                            try {
                                                HfTokenStore.save(context, tokenForDownload)

                                                withContext(Dispatchers.IO) {

                                                    xttsModels.downloadAndInstall(
                                                        tokenForDownload
                                                    ) { value ->
                                                        if (ProgressLimiter.allow("xtts-download", 120L) || value <= 0f || value >= 1f) {
                                                            scope.launch(Dispatchers.Main.immediate) {
                                                                progress = value.coerceIn(0f, 1f)
                                                                progressEta = estimateRemainingLabel(value, startedAt)
                                                            }
                                                        }
                                                    }
                                                }

                                                xttsInstalled = true

                                                xttsModelStatus =
                                                    "Model gotowy"

                                                status =
                                                    "Pakiet XTTS-v2 jest gotowy do testu grafów ONNX."

                                            } catch (e: Exception) {

                                                xttsModelStatus =
                                                    "Błąd pobierania: " +
                                                    (e.message ?: e::class.java.simpleName)

                                            } finally {
                                                busy = false
                                            }
                                        }
                                    },
                                    enabled = !busy && hfToken.trim().startsWith("hf_"),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text(
                                        if (busy) "Pobieram model…"
                                        else if (xttsInstalled)
                                            "Pobierz ponownie model XTTS"
                                        else
                                            "Pobierz i zainstaluj XTTS (INT8)"
                                    )
                                }

                                if (xttsInstalled) {
                                    OutlinedButton(onClick = { pendingModelDelete = "xtts" }, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                                        Text(tr("Usuń model XTTS-v2", "Delete XTTS-v2 model"))
                                    }

                                    OutlinedButton(
                                        onClick = {

                                            scope.launch {

                                                busy = true
                                                xttsGraphInfo = ""

                                                xttsModelStatus =
                                                    "Otwieram grafy ONNX i sprawdzam ich wejścia/wyjścia…"

                                                try {

                                                    val report = withContext(Dispatchers.IO) {

                                                        val sessions =
                                                            XtTsOnnxRuntime.openSessions(
                                                                xttsModels.modelDir(),
                                                                numThreads = 2
                                                            )

                                                        try {

                                                            val inputs =
                                                                sessions.inputNames()
                                                                    .entries
                                                                    .joinToString("\n") {
                                                                        "${it.key}: ${it.value.joinToString()}"
                                                                    }

                                                            val outputs =
                                                                sessions.outputNames()
                                                                    .entries
                                                                    .joinToString("\n") {
                                                                        "${it.key}: ${it.value.joinToString()}"
                                                                    }

                                                            "WEJŚCIA\n$inputs\n\nWYJŚCIA\n$outputs"

                                                        } finally {

                                                            sessions.close()
                                                        }
                                                    }

                                                    xttsGraphInfo = report

                                                    xttsModelStatus =
                                                        "Wszystkie cztery grafy ONNX zostały otwarte i zamknięte poprawnie."

                                                } catch (e: Exception) {

                                                    xttsModelStatus =
                                                        "Test ONNX nie powiódł się: " +
                                                        (e.message ?: e::class.java.simpleName)

                                                } finally {

                                                    busy = false
                                                }
                                            }
                                        },
                                        enabled = !busy,
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Text(tr("Sprawdź uruchomienie grafów ONNX", "Test ONNX graph execution"))
                                    }
                                }

                                Text(
                                    xttsModelStatus,
                                    style = MaterialTheme.typography.bodySmall
                                )

                                if (xttsGraphInfo.isNotBlank()) {

                                    Text(
                                        xttsGraphInfo,
                                        style = MaterialTheme.typography.bodySmall
                                    )
                                }

                                Text(
                                    tr("Wagi tego repozytorium są objęte CC BY-NC 4.0 — użytek niekomercyjny. ", "The model weights in this repository are licensed under CC BY-NC 4.0 for non-commercial use. ") +
                                    tr("Używaj wyłącznie własnego głosu albo nagrań z wyraźną zgodą właściciela.", "Use only your own voice or recordings with the owner’s explicit permission."),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        if (busy) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                    }

                    Screen.Settings -> {
                        Text(tr("Ustawienia", "Settings"), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

                        Card {
                            Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text(tr("Aplikacja", "Application"), style = MaterialTheme.typography.titleMedium)
                                Text(tr("Język interfejsu", "Interface language"))
                                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                                    listOf("pl" to "Polski", "en" to "English").forEachIndexed { index, option ->
                                        SegmentedButton(
                                            selected = appLanguage == option.first,
                                            onClick = { selectAppLanguage(option.first) },
                                            shape = SegmentedButtonDefaults.itemShape(index, 2)
                                        ) { Text(option.second) }
                                    }
                                }
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                                        Text(tr("Automatyczna synchronizacja języka", "Automatic language synchronization"), fontWeight = FontWeight.Medium)
                                        Text(tr("Zmiana języka aplikacji ustawia język syntezy obu silników. Ręczny wybór języka wyłącza tę opcję.", "Changing the app language updates both synthesis engines. Choosing a synthesis language manually disables this option."), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                    Switch(checked = autoSyncLanguage, onCheckedChange = { setAutoSyncLanguage(it) })
                                }
                            }
                        }

                        Card {
                            Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
                                Text("XTTS-v2", style = MaterialTheme.typography.titleMedium)
                                Text(tr("Ustawienia generowania", "Generation settings"), style = MaterialTheme.typography.titleSmall)
                                Text("${tr("Tempo", "Speed")}: ${formatNumber(xttsSpeed)}×")
                                Slider(value = xttsSpeed, onValueChange = { xttsSpeed = it; xttsPreferences.edit().putFloat("xtts_speed", it).apply() }, valueRange = 0.5f..2.0f)
                                Text("Temperature: ${formatNumber(xttsTemperature)}")
                                Slider(value = xttsTemperature, onValueChange = { xttsTemperature = it; xttsPreferences.edit().putFloat("xtts_temperature", it).apply() }, valueRange = 0.1f..2.0f)
                                Text("Top-K: $xttsTopK")
                                Slider(value = xttsTopK.toFloat(), onValueChange = { xttsTopK = it.toInt().coerceIn(1, 100); xttsPreferences.edit().putInt("xtts_top_k", xttsTopK).apply() }, valueRange = 1f..100f, steps = 98)
                                Text("Top-P: ${formatNumber(xttsTopP)}")
                                Slider(value = xttsTopP, onValueChange = { xttsTopP = it; xttsPreferences.edit().putFloat("xtts_top_p", it).apply() }, valueRange = 0.1f..1.0f)
                                Text("Repetition penalty: ${formatNumber(xttsRepetitionPenalty)}")
                                Slider(value = xttsRepetitionPenalty, onValueChange = { xttsRepetitionPenalty = it; xttsPreferences.edit().putFloat("xtts_repetition_penalty", it).apply() }, valueRange = 1.0f..2.0f)
                                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                Text(tr("Język syntezy XTTS", "XTTS synthesis language"))
                                var xttsSettingsLangExpanded by remember { mutableStateOf(false) }
                                ExposedDropdownMenuBox(expanded = xttsSettingsLangExpanded, onExpandedChange = { xttsSettingsLangExpanded = it }) {
                                    OutlinedTextField(value = xttsLanguage, onValueChange = {}, readOnly = true, modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text(tr("Kod języka", "Language code")) })
                                    ExposedDropdownMenu(expanded = xttsSettingsLangExpanded, onDismissRequest = { xttsSettingsLangExpanded = false }) {
                                        listOf("pl", "en", "de", "fr", "es", "it", "pt", "cs", "ru", "nl", "tr", "zh", "ja", "hu", "ko", "ar", "hi").forEach { lang ->
                                            DropdownMenuItem(text = { Text(lang) }, onClick = { selectXttsLanguage(lang); xttsSettingsLangExpanded = false })
                                        }
                                    }
                                }
                            }
                        }

                        Card {
                            Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
                                Text("Supertonic 3", style = MaterialTheme.typography.titleMedium)
                                Text(tr("Ustawienia generowania", "Generation settings"), style = MaterialTheme.typography.titleSmall)
                                Text("${tr("Tempo", "Speed")}: ${formatNumber(supertonicSpeed)}×")
                                Slider(value = supertonicSpeed, onValueChange = { supertonicSpeed = it; xttsPreferences.edit().putFloat("supertonic_speed", it).apply() }, valueRange = 0.6f..1.6f)
                                Text("${tr("Kroki", "Steps")}: $steps")
                                Slider(value = steps.toFloat(), onValueChange = { steps = it.toInt().coerceIn(4, 128); xttsPreferences.edit().putInt("supertonic_steps", steps).apply() }, valueRange = 4f..128f, steps = 123)
                                Text("${tr("Głos", "Voice")}: ${sid + 1} / 10")
                                Slider(value = sid.toFloat(), onValueChange = { sid = it.toInt().coerceIn(0, 9); xttsPreferences.edit().putInt("supertonic_sid", sid).apply() }, valueRange = 0f..9f, steps = 8)
                                Text("${tr("Wątki CPU", "CPU threads")}: $supertonicThreads")
                                Slider(value = supertonicThreads.toFloat(), onValueChange = { supertonicThreads = it.toInt().coerceIn(1, 8); xttsPreferences.edit().putInt("supertonic_threads", supertonicThreads).apply() }, onValueChangeFinished = { if (!busy) supertonic.close() }, enabled = !busy, valueRange = 1f..8f, steps = 6)
                                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                Text(tr("Język syntezy Supertonic", "Supertonic synthesis language"))
                                var supertonicSettingsLangExpanded by remember { mutableStateOf(false) }
                                ExposedDropdownMenuBox(expanded = supertonicSettingsLangExpanded, onExpandedChange = { supertonicSettingsLangExpanded = it }) {
                                    OutlinedTextField(value = supertonicLanguage, onValueChange = {}, readOnly = true, modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text(tr("Kod języka", "Language code")) })
                                    ExposedDropdownMenu(expanded = supertonicSettingsLangExpanded, onDismissRequest = { supertonicSettingsLangExpanded = false }) {
                                        SupertonicEngine.SUPPORTED_LANGUAGES.sorted().forEach { lang ->
                                            DropdownMenuItem(text = { Text(lang) }, onClick = { selectSupertonicLanguage(lang); supertonicSettingsLangExpanded = false })
                                        }
                                    }
                                }
                            }
                        }

                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text(tr("Trwały magazyn danych", "Persistent storage"), style = MaterialTheme.typography.titleMedium)
                                Text(tr("Modele, profile głosu i wygenerowane WAV są przechowywane w Documents/KinoTTS. Nie są trzymane w katalogu aplikacji.", "Models, voice profiles, and generated WAV files are stored in Documents/KinoTTS, not inside the app directory."))
                                Text(if (KinoStorage.hasAccess(context)) "Dostęp do Documents: przyznany" else "Dostęp do Documents: wymagany")
                                OutlinedButton(
                                    onClick = { KinoStorage.requestAccess(context) },
                                    modifier = Modifier.fillMaxWidth()
                                ) { Text(tr("Nadaj / sprawdź dostęp do Documents", "Grant / check Documents access")) }
                                Text(tr("Android nie usuwa Documents/KinoTTS przy odinstalowaniu aplikacji. Usunięcie tych plików wymaga ręcznego skasowania folderu w menedżerze plików.", "Android does not remove Documents/KinoTTS when the app is uninstalled. Delete these files manually in your file manager."), style = MaterialTheme.typography.bodySmall)
                                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                Text(tr("Przetwarzanie", "Processing"), style = MaterialTheme.typography.titleMedium)
                                Text(tr("XTTS: ${xttsRuntime.executionProviderStatus}", "XTTS: ${xttsRuntime.executionProviderStatus}"))
                                Text(tr("Backend ostatniej syntezy: $computeBackendStatus", "Last synthesis backend: $computeBackendStatus"))
                                Text(ramStats)
                                Text(tr("ONNX Runtime: $ortStatus", "ONNX Runtime: $ortStatus"))
                                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                Text(tr("Backend XTTS", "XTTS backend"), style = MaterialTheme.typography.titleSmall)
                                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                                    listOf("cpu" to "CPU", "nnapi" to "NNAPI").forEachIndexed { index, option ->
                                        SegmentedButton(
                                            selected = xttsExecutionProvider == option.first,
                                            onClick = {
                                                if (!busy && xttsExecutionProvider != option.first) {
                                                    scope.launch {
                                                        busy = true
                                                        try {
                                                            withContext(Dispatchers.IO) {
                                                                xttsRuntime.setExecutionProvider(option.first)
                                                            }
                                                            xttsExecutionProvider = option.first
                                                            xttsPreferences.edit().putString("xtts_execution_provider", option.first).apply()
                                                            status = if (option.first == "nnapi") "Wybrano eksperymentalny backend NNAPI dla XTTS." else "Wybrano backend CPU dla XTTS."
                                                        } catch (e: Exception) {
                                                            status = "Nie udało się zmienić backendu: ${e.message ?: e::class.java.simpleName}"
                                                        } finally {
                                                            busy = false
                                                        }
                                                    }
                                                }
                                            },
                                            enabled = !busy,
                                            shape = SegmentedButtonDefaults.itemShape(index, 2)
                                        ) { Text(option.second) }
                                    }
                                }
                                Text(
                                    "CPU oznacza wykonanie przez CPU. NNAPI oznacza, że ONNX Runtime przekazał obsługiwane operatory do Androida; sterownik może użyć CPU, GPU lub NPU, ale ta wersja ORT nie raportuje osobno rzeczywistego urządzenia dla każdej warstwy. Po błędzie NNAPI aplikacja pokazuje CPU fallback.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                if (ramWarning) Text(
                                    "Mało dostępnej pamięci RAM. Zamknij zbędne aplikacje.",
                                    color = MaterialTheme.colorScheme.error
                                )
                                Text(tr("Po pobraniu model Supertonic generuje offline.", "After downloading the model, Supertonic works offline."))
                                Text(tr("Supertonic 3: obecnie jawnie skonfigurowany provider CPU. XTTS: CPU albo NNAPI; GPU/NPU nie będą zgłaszane jako aktywne bez potwierdzenia przez runtime.", "Supertonic 3 currently uses the explicitly configured CPU provider. XTTS supports CPU or NNAPI; GPU/NPU are not reported as active unless confirmed by the runtime."))
                            }
                        }
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(tr("Prywatność i zgoda", "Privacy and consent"), style = MaterialTheme.typography.titleMedium)
                                Text(tr("Próbki profili są zapisywane lokalnie. Zapis jest dostępny po potwierdzeniu, że to Twój głos albo masz wyraźną zgodę właściciela nagrania.", "Voice samples are stored locally. Saving is available after confirming that this is your voice or that you have the recording owner’s explicit permission."))
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun VoiceProfileCard(profile: VoiceProfile, english: Boolean, onDelete: () -> Unit, enabled: Boolean) {
    Card {
        Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(profile.name, style = MaterialTheme.typography.titleMedium)
            Text(if (english) "${formatSeconds(profile.durationSeconds)} • ${profile.sampleRate} Hz • language: ${profile.language}" else "${formatSeconds(profile.durationSeconds)} • ${profile.sampleRate} Hz • język: ${profile.language}")
            Text(
                "RMS ${formatNumber(profile.rmsDbfs)} dBFS • clipping ${formatNumber(profile.clippingPercent)}% • cisza ${formatNumber(profile.silencePercent)}%",
                style = MaterialTheme.typography.bodySmall
            )
            OutlinedButton(onClick = onDelete, enabled = enabled, modifier = Modifier.fillMaxWidth()) {
                Text(if (english) "Delete sample" else "Usuń próbkę")
            }
        }
    }
}

@Composable
private fun StatusCard(status: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Row(Modifier.fillMaxWidth().padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.Tune, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text(status, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

private fun defaultPrompt(language: String): String =
    if (language == "pl") "Cześć! To jest test syntezy mowy w KinoTTS Studio."
    else "Hello! This is a speech synthesis test in KinoTTS Studio."

private fun screenTitle(screen: Screen, english: Boolean): String = when (screen) {
    Screen.Studio -> "Studio"
    Screen.Voices -> if (english) "Voices" else "Głosy"
    Screen.Models -> if (english) "Models" else "Modele"
    Screen.Settings -> if (english) "Settings" else "Ustawienia"
}

private fun localizeStatusEnglish(status: String): String {
    val exact = mapOf(
        "Gotowy" to "Ready",
        "Zapisano plik WAV." to "WAV file saved.",
        "Usunięto model XTTS-v2." to "XTTS-v2 model deleted.",
        "Usunięto model Supertonic 3." to "Supertonic 3 model deleted.",
        "Próbka gotowa. Zapisz ją jako profil głosu." to "Sample ready. Save it as a voice profile.",
        "Dekoduję próbkę audio…" to "Decoding audio sample…",
        "Uruchamiam XTTS-v2…" to "Starting XTTS-v2…",
        "Wysłano żądanie anulowania XTTS…" to "XTTS cancellation requested…",
        "Sprawdzam model Supertonic 3…" to "Checking Supertonic 3 model…",
        "Ładuję model i generuję mowę…" to "Loading model and synthesizing speech…",
        "Za mało pamięci dla syntezy Supertonic. Skróć tekst i zamknij aplikacje działające w tle." to "Not enough memory for Supertonic synthesis. Shorten the text and close background apps.",
        "Model Supertonic 3 jest gotowy" to "Supertonic 3 model is ready",
        "Token Hugging Face zapisany bezpiecznie na tym urządzeniu." to "Hugging Face token saved securely on this device.",
        "Usunięto zapisany token Hugging Face." to "Saved Hugging Face token deleted.",
        "Pakiet XTTS-v2 jest gotowy do testu grafów ONNX." to "XTTS-v2 package is ready for ONNX graph testing.",
        "Plik wybrany. Możesz wykonać lokalną analizę audio." to "File selected. You can run local audio analysis.",
        "Wybrano eksperymentalny backend NNAPI dla XTTS." to "Experimental NNAPI backend selected for XTTS.",
        "Wybrano backend CPU dla XTTS." to "CPU backend selected for XTTS."
    )
    exact[status]?.let { return it }
    val prefixes = listOf(
        "Błąd zapisu: " to "Save error: ",
        "Błąd usuwania modelu: " to "Model deletion error: ",
        "Przełączam silnik na " to "Switching engine to ",
        "Przełączono silnik: " to "Engine switched: ",
        "Nie udało się przełączyć silnika: " to "Could not switch engine: ",
        "Błąd analizy: " to "Analysis error: ",
        "Błąd zapisu profilu: " to "Voice profile save error: ",
        "XTTS wygenerował WAV: " to "XTTS generated WAV: ",
        "Błąd XTTS: " to "XTTS error: ",
        "Błąd biblioteki natywnej Supertonic: " to "Supertonic native library error: ",
        "Błąd syntezy: " to "Synthesis error: ",
        "Gotowe: " to "Done: ",
        "Usunięto próbkę z głosu " to "Sample removed from voice ",
        "Nie udało się usunąć próbki: " to "Could not delete sample: ",
        "Błąd pobierania: " to "Download error: ",
        "Nie udało się zapisać tokena: " to "Could not save token: ",
        "Nie udało się zmienić backendu: " to "Could not change backend: ",
        "Utworzono głos " to "Created voice ",
        "Dodano próbkę do głosu " to "Added a sample to voice "
    )
    for ((polish, english) in prefixes) {
        if (status.startsWith(polish)) return english + status.removePrefix(polish)
    }
    return status
}

private object ProgressLimiter {
    private val stamps = java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.atomic.AtomicLong>()
    fun allow(key: String, intervalMs: Long): Boolean {
        val now = android.os.SystemClock.elapsedRealtime()
        val stamp = stamps.getOrPut(key) { java.util.concurrent.atomic.AtomicLong(0L) }
        while (true) {
            val old = stamp.get()
            if (now - old < intervalMs) return false
            if (stamp.compareAndSet(old, now)) return true
        }
    }
}

private fun formatNumber(value: Float): String = String.format(Locale.US, "%.1f", value)
private fun formatSeconds(value: Double): String = String.format(Locale.US, "%.1f s", value)


private fun estimateRemainingLabel(progress: Float, startedAt: Long): String {
    val p = progress.coerceIn(0f, 1f)
    if (p < 0.02f) return "Szacowany czas: obliczanie…"
    if (p >= 0.999f) return "Pozostało: 0 s"

    val elapsedMs = (System.currentTimeMillis() - startedAt).coerceAtLeast(1L)
    val remainingSeconds = (elapsedMs.toDouble() * (1.0 - p) / p / 1000.0)
        .toLong().coerceAtLeast(0L)

    val minutes = remainingSeconds / 60
    val seconds = remainingSeconds % 60
    return if (minutes > 0) {
        "Szacowany czas: około ${minutes} min ${seconds} s"
    } else {
        "Szacowany czas: około ${seconds} s"
    }
}
