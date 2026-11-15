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
    val analyzer = remember { AudioAnalyzer(context.applicationContext) }
    val profileStore = remember { VoiceProfileStore(context.applicationContext) }
    val supertonic = remember { SupertonicEngine(KinoStorage.outputDir(context).apply { mkdirs() }) }
    val xttsRuntime = remember { XttsSynthesizer(xttsModels.modelDir(), KinoStorage.outputDir(context).apply { mkdirs() }) }

    var screen by remember { mutableStateOf(Screen.Studio) }
    var engine by remember { mutableStateOf("Supertonic 3") }
    var text by remember { mutableStateOf("Cześć! To jest test syntezy mowy w KinoTTS Studio.") }
    var selectedAudio by remember { mutableStateOf<Uri?>(null) }
    var selectedAudioName by remember { mutableStateOf("Nie wybrano pliku") }
    var analysis by remember { mutableStateOf<AudioAnalysisResult?>(null) }
    var profileName by remember { mutableStateOf("Mój głos") }
    var consentConfirmed by remember { mutableStateOf(false) }
    var profiles by remember { mutableStateOf(profileStore.list()) }
    var xttsChosenProfileId by remember { mutableStateOf("") }
    var xttsSynthesizing by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("Gotowy") }
    var hfToken by remember { mutableStateOf(HfTokenStore.load(context).orEmpty()) }
    var xttsInstalled by remember { mutableStateOf(xttsModels.isInstalled()) }
    var xttsModelStatus by remember {
        mutableStateOf(if (xttsInstalled) "Model XTTS jest zainstalowany." else "Model XTTS nie jest jeszcze pobrany.")
    }
    var supertonicModelStatus by remember {
        mutableStateOf(if (models.isSupertonicInstalled()) "Model jest zainstalowany." else "Model nie jest jeszcze pobrany.")
    }
    var xttsGraphInfo by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
    var progressEta by remember { mutableStateOf("") }
    var sid by remember { mutableIntStateOf(0) }
    var speed by remember { mutableFloatStateOf(1f) }
    var steps by remember { mutableIntStateOf(8) }
    var customStepsEnabled by remember { mutableStateOf(false) }
    var customStepsText by remember { mutableStateOf("160") }
    val customStepValue = customStepsText.toIntOrNull()
    val effectiveSteps = if (customStepsEnabled) customStepValue ?: steps else steps
    val stepsValid = !customStepsEnabled ||
        (customStepValue != null && customStepValue in 129..256)
    var ramStats by remember { mutableStateOf("Odczyt pamięci…") }
    var ramWarning by remember { mutableStateOf(false) }
    var ortStatus by remember { mutableStateOf("Sprawdzanie ONNX Runtime…") }
    var language by remember { mutableStateOf("pl") }
    var output by remember { mutableStateOf<File?>(null) }
    var player by remember { mutableStateOf<MediaPlayer?>(null) }

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

    LaunchedEffect(Unit) {
        ortStatus = withContext(Dispatchers.IO) {
            XtTsOnnxRuntime.diagnose().message
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
                                Text("XTTS-v2", fontWeight = FontWeight.Bold)
                                Spacer(Modifier.width(8.dp))
                                Text(
                                    xttsModelStatus,
                                    modifier = Modifier.weight(1f),
                                    style = MaterialTheme.typography.bodySmall,
                                    maxLines = 2
                                )
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("Supertonic 3", fontWeight = FontWeight.Bold)
                                Spacer(Modifier.width(8.dp))
                                Text(
                                    supertonicModelStatus,
                                    modifier = Modifier.weight(1f),
                                    style = MaterialTheme.typography.bodySmall,
                                    maxLines = 2
                                )
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
                                icon = { Icon(icon, contentDescription = item.title) },
                                label = { Text(item.title) }
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
                        Text("Generator mowy", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                Text("Silnik", style = MaterialTheme.typography.titleMedium)
                                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                                    listOf("XTTS-v2", "Supertonic 3").forEachIndexed { index, name ->
                                        SegmentedButton(
                                            selected = engine == name,
                                            onClick = { engine = name },
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
                                        Spacer(Modifier.width(8.dp)); Text("Wybierz MP3 / MP4 / WAV")
                                    }
                                    Text(selectedAudioName, style = MaterialTheme.typography.bodySmall)
                                    Button(
                                        onClick = {
                                            val uri = selectedAudio
                                            if (uri != null) scope.launch {
                                                busy = true; progress = 0f; status = "Dekoduję próbkę audio…"
                                                try {
                                                    analysis = withContext(Dispatchers.IO) {
                                                        analyzer.analyze(uri) { value -> scope.launch { progress = value } }
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
                                                Text("Parametry próbki", style = MaterialTheme.typography.titleMedium)
                                                Text("Czas: ${formatSeconds(result.durationSeconds)} • ${result.sampleRate} Hz • ${result.sourceChannels} kan.")
                                                Text("RMS: ${formatNumber(result.rmsDbfs)} dBFS • clipping: ${formatNumber(result.clippingPercent)}% • cisza: ${formatNumber(result.silencePercent)}%")
                                            }
                                        }
                                        OutlinedTextField(
                                            value = profileName, onValueChange = { profileName = it },
                                            label = { Text("Nazwa profilu głosu") }, modifier = Modifier.fillMaxWidth(), singleLine = true
                                        )
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Checkbox(checked = consentConfirmed, onCheckedChange = { consentConfirmed = it })
                                            Text("To mój głos albo mam wyraźną zgodę właściciela nagrania.", style = MaterialTheme.typography.bodySmall)
                                        }
                                        Button(
                                            onClick = {
                                                scope.launch {
                                                    busy = true
                                                    try {
                                                        val saved = withContext(Dispatchers.IO) {
                                                            profileStore.save(profileName, result, language, consentConfirmed)
                                                        }
                                                        profiles = withContext(Dispatchers.IO) { profileStore.list() }
                                                        xttsChosenProfileId = saved.id
                                                        consentConfirmed = false
                                                        status = "Profil ${saved.name} zapisany. Możesz go wybrać do klonowania głosu."
                                                    } catch (e: Exception) {
                                                        status = "Błąd zapisu profilu: ${e.message ?: e::class.java.simpleName}"
                                                    } finally { busy = false }
                                                }
                                            }, enabled = !busy && consentConfirmed && profileName.trim().length in 2..40,
                                            modifier = Modifier.fillMaxWidth()
                                        ) { Text("Zapisz profil głosu") }
                                    }

                                    Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                    Text("Tekst do wygenerowania", style = MaterialTheme.typography.titleMedium)
                                    OutlinedTextField(
                                        value = text, onValueChange = { text = it }, label = { Text("Tekst po polsku lub w innym języku XTTS") },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp), minLines = 3
                                    )
                                    var xttsLanguageExpanded by remember { mutableStateOf(false) }
                                    ExposedDropdownMenuBox(expanded = xttsLanguageExpanded, onExpandedChange = { xttsLanguageExpanded = it }) {
                                        OutlinedTextField(
                                            value = language, onValueChange = {}, readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text("Język XTTS") }
                                        )
                                        ExposedDropdownMenu(expanded = xttsLanguageExpanded, onDismissRequest = { xttsLanguageExpanded = false }) {
                                            listOf("pl", "en", "de", "fr", "es", "it", "pt", "cs", "ru", "nl", "tr", "zh", "ja", "hu", "ko", "ar", "hi").forEach { lang ->
                                                DropdownMenuItem(text = { Text(lang) }, onClick = { language = lang; xttsLanguageExpanded = false })
                                            }
                                        }
                                    }
                                    val selectedXttsProfile = profiles.firstOrNull { it.id == xttsChosenProfileId } ?: profiles.firstOrNull()
                                    var xttsProfileExpanded by remember { mutableStateOf(false) }
                                    ExposedDropdownMenuBox(expanded = xttsProfileExpanded, onExpandedChange = { xttsProfileExpanded = it }) {
                                        OutlinedTextField(
                                            value = selectedXttsProfile?.name ?: "Brak zapisanego profilu",
                                            onValueChange = {}, readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text("Głos do sklonowania") }
                                        )
                                        ExposedDropdownMenu(expanded = xttsProfileExpanded, onDismissRequest = { xttsProfileExpanded = false }) {
                                            profiles.forEach { profile ->
                                                DropdownMenuItem(text = { Text(profile.name) }, onClick = {
                                                    xttsChosenProfileId = profile.id; xttsProfileExpanded = false
                                                })
                                            }
                                        }
                                    }
                                    Text("Tempo: ${formatNumber(speed)}×")
                                    Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.7f..1.4f)
                                    if (!xttsInstalled) {
                                        Text("Najpierw pobierz XTTS-v2 w zakładce Modele.", color = MaterialTheme.colorScheme.error)
                                    }
                                    if (profiles.isEmpty()) {
                                        Text("Najpierw zaimportuj próbkę i zapisz profil głosu.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                    Button(
                                        onClick = {
                                            val selected = selectedXttsProfile
                                            if (selected != null) scope.launch {
                                                busy = true; xttsSynthesizing = true; progress = 0f; status = "Uruchamiam XTTS-v2…"
                                                try {
                                                    val wav = withContext(Dispatchers.IO) {
                                                        xttsRuntime.synthesize(text, language, File(selected.wavPath), speed) { value ->
                                                            scope.launch { progress = value }
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
                                    ) { Text(if (busy) "Generuję głos…" else "Generuj mowę XTTS") }
                                    if (xttsSynthesizing) {
                                        OutlinedButton(
                                            onClick = { xttsRuntime.cancel(); status = "Wysłano żądanie anulowania XTTS…" },
                                            modifier = Modifier.fillMaxWidth()
                                        ) { Text("Anuluj generowanie") }
                                    }
                                } else {
                                    OutlinedTextField(
                                        value = text,
                                        onValueChange = { text = it },
                                        label = { Text("Tekst do przeczytania") },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 140.dp),
                                        minLines = 4
                                    )
                                    Text("Głos: ${sid + 1} / 10")
                                    Slider(value = sid.toFloat(), onValueChange = { sid = it.toInt() }, valueRange = 0f..9f, steps = 8)
                                    Text("Język")
                                    var languageExpanded by remember { mutableStateOf(false) }
                                    ExposedDropdownMenuBox(expanded = languageExpanded, onExpandedChange = { languageExpanded = it }) {
                                        OutlinedTextField(
                                            value = language,
                                            onValueChange = {},
                                            readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(),
                                            label = { Text("Kod języka") }
                                        )
                                        ExposedDropdownMenu(expanded = languageExpanded, onDismissRequest = { languageExpanded = false }) {
                                            listOf("pl", "en", "de", "fr", "es", "it", "cs", "uk", "nl", "pt", "sv").forEach { lang ->
                                                DropdownMenuItem(
                                                    text = { Text(lang) },
                                                    onClick = { language = lang; languageExpanded = false }
                                                )
                                            }
                                        }
                                    }
                                    Text("Tempo: ${formatNumber(speed)}×")
                                    Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.7f..1.4f)
                                    Text("Kroki: ${if (customStepsEnabled) effectiveSteps else steps}")
                                    Slider(value = steps.toFloat(),
                                        onValueChange = { steps = it.toInt() },
                                        valueRange = 4f..128f, steps = 123)
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Checkbox(customStepsEnabled, { customStepsEnabled = it })
                                        Text("Custom — powyżej 128")
                                    }
                                    if (customStepsEnabled) {
                                        OutlinedTextField(
                                            value = customStepsText,
                                            onValueChange = { customStepsText = it.filter(Char::isDigit).take(3) },
                                            label = { Text("Kroki (129–256)") },
                                            isError = !stepsValid,
                                            supportingText = {
                                                Text(if (stepsValid) "Więcej kroków oznacza dłuższe generowanie."
                                                    else "Wpisz wartość od 129 do 256.")
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            singleLine = true
                                        )
                                    }
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
                                                models.ensureSupertonic { value -> progress = value }
                                            }
                                            status = "Ładuję model i generuję mowę…"
                                            val wav = withContext(Dispatchers.IO) {
                                                supertonic.load(modelDir)
                                                supertonic.generate(text, sid, speed, effectiveSteps, language)
                                            }
                                            output = wav
                                            status = "Gotowe: ${wav.name}"
                                        } catch (e: Exception) {
                                            status = "Błąd syntezy: ${e.message ?: e::class.java.simpleName}"
                                        } finally {
                                            busy = false
                                        }
                                    }
                                },
                                enabled = !busy && text.isNotBlank() && stepsValid,
                                modifier = Modifier.fillMaxWidth()
                            ) { Text(if (busy) "Pracuję…" else "Generuj mowę") }
                        }

                        if (busy) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                        StatusCard(status)
                        output?.let { file ->
                            Card {
                                Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text("Wynik audio", style = MaterialTheme.typography.titleMedium)
                                    Text(file.name, style = MaterialTheme.typography.bodySmall)
                                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        OutlinedButton(onClick = {
                                            player?.release()
                                            player = MediaPlayer().apply {
                                                setDataSource(file.absolutePath)
                                                prepare()
                                                start()
                                            }
                                        }) { Text("Odtwórz") }
                                        OutlinedButton(onClick = { exportAudio.launch(file.name) }) {
                                            Text("Zapisz WAV")
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Screen.Voices -> {
                        Text("Biblioteka głosów", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Supertonic 3", style = MaterialTheme.typography.titleMedium)
                                Text("10 gotowych stylów głosu (ID 0–9). Wybierz głos w Studio.")
                            }
                        }
                        Text("Zapisane próbki", style = MaterialTheme.typography.titleMedium)
                        if (profiles.isEmpty()) {
                            Text("Nie zapisano jeszcze żadnej próbki.", style = MaterialTheme.typography.bodySmall)
                        } else {
                            profiles.forEach { profile -> VoiceProfileCard(profile) }
                        }
                        Text(
                            "Zapisane próbki mogą służyć jako profile referencyjne dla lokalnego silnika XTTS-v2.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Screen.Models -> {
                        Text("Menedżer modeli", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
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
                                                        progress = value
                                                        progressEta = estimateRemainingLabel(value, startedAt)
                                                    }
                                                }
                                                supertonicModelStatus = "Model gotowy — zapisany w Documents/KinoTTS/Models"
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
                                ) { Text("Pobierz / sprawdź model") }
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
                                        "Model pobrany i zainstalowany w Documents/KinoTTS/Models."
                                    else
                                        "Pobierz model z Hugging Face. Po instalacji pliki pozostaną na telefonie."
                                )

                                OutlinedTextField(
                                    value = hfToken,
                                    onValueChange = { hfToken = it },
                                    label = { Text("Hugging Face token (hf_… )") },
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
                                                        progress = value
                                                        progressEta = estimateRemainingLabel(value, startedAt)
                                                    }
                                                }

                                                xttsInstalled = true

                                                xttsModelStatus =
                                                    "Model XTTS-v2 został pobrany, sprawdzony i zainstalowany."

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
                                        Text("Sprawdź uruchomienie grafów ONNX")
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
                                    "Wagi tego repozytorium są objęte CC BY-NC 4.0 — użytek niekomercyjny. " +
                                    "Używaj wyłącznie własnego głosu albo nagrań z wyraźną zgodą właściciela.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        if (busy) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                        StatusCard(status)
                    }

                    Screen.Settings -> {
                        Text("Ustawienia", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text("Trwały magazyn danych", style = MaterialTheme.typography.titleMedium)
                                Text("Modele, profile głosu i wygenerowane WAV są przechowywane w Documents/KinoTTS. Nie są trzymane w katalogu aplikacji.")
                                Text(if (KinoStorage.hasAccess(context)) "Dostęp do Documents: przyznany" else "Dostęp do Documents: wymagany")
                                OutlinedButton(
                                    onClick = { KinoStorage.requestAccess(context) },
                                    modifier = Modifier.fillMaxWidth()
                                ) { Text("Nadaj / sprawdź dostęp do Documents") }
                                Text("Android nie usuwa Documents/KinoTTS przy odinstalowaniu aplikacji. Usunięcie tych plików wymaga ręcznego skasowania folderu w menedżerze plików.", style = MaterialTheme.typography.bodySmall)
                                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                                Text("Przetwarzanie", style = MaterialTheme.typography.titleMedium)
                                Text("Silnik: CPU • ARM64 • modele w Documents/KinoTTS")
                                Text(ramStats)
                                Text("ONNX Runtime: $ortStatus")
                                if (ramWarning) Text(
                                    "Mało dostępnej pamięci RAM. Zamknij zbędne aplikacje.",
                                    color = MaterialTheme.colorScheme.error
                                )
                                Text("Po pobraniu model Supertonic generuje offline.")
                                Text("NNAPI/NPU: obsługa i przyspieszenie sprzętowe wymagają osobnego testu na tym urządzeniu.")
                            }
                        }
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Prywatność i zgoda", style = MaterialTheme.typography.titleMedium)
                                Text("Próbki profili są zapisywane lokalnie. Zapis jest dostępny po potwierdzeniu, że to Twój głos albo masz wyraźną zgodę właściciela nagrania.")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun VoiceProfileCard(profile: VoiceProfile) {
    Card {
        Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(profile.name, style = MaterialTheme.typography.titleMedium)
            Text("${formatSeconds(profile.durationSeconds)} • ${profile.sampleRate} Hz • język: ${profile.language}")
            Text(
                "RMS ${formatNumber(profile.rmsDbfs)} dBFS • clipping ${formatNumber(profile.clippingPercent)}% • cisza ${formatNumber(profile.silencePercent)}%",
                style = MaterialTheme.typography.bodySmall
            )
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
