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
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.kinotts.studio.analysis.AudioAnalysisResult
import pl.kinotts.studio.analysis.AudioAnalyzer
import pl.kinotts.studio.engine.SupertonicEngine
import pl.kinotts.studio.models.ModelManager
import pl.kinotts.studio.voices.VoiceProfile
import pl.kinotts.studio.voices.VoiceProfileStore
import java.io.File
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { KinoTtsApp() }
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
    val analyzer = remember { AudioAnalyzer(context.applicationContext) }
    val profileStore = remember { VoiceProfileStore(context.applicationContext) }
    val supertonic = remember { SupertonicEngine(File(context.filesDir, "audio")) }

    var screen by remember { mutableStateOf(Screen.Studio) }
    var engine by remember { mutableStateOf("Supertonic 3") }
    var text by remember { mutableStateOf("Cześć! To jest test syntezy mowy w KinoTTS Studio.") }
    var selectedAudio by remember { mutableStateOf<Uri?>(null) }
    var selectedAudioName by remember { mutableStateOf("Nie wybrano pliku") }
    var analysis by remember { mutableStateOf<AudioAnalysisResult?>(null) }
    var profileName by remember { mutableStateOf("Mój głos") }
    var consentConfirmed by remember { mutableStateOf(false) }
    var profiles by remember { mutableStateOf(profileStore.list()) }
    var status by remember { mutableStateOf("Gotowy") }
    var busy by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
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

    DisposableEffect(Unit) {
        onDispose {
            player?.release()
            supertonic.close()
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
                                        "XTTS-v2: synteza i klonowanie nie są jeszcze podłączone. Poniżej działa przygotowanie próbki: dekodowanie, analiza jakości i zapis profilu w pamięci aplikacji.",
                                        color = MaterialTheme.colorScheme.error
                                    )
                                    OutlinedButton(
                                        onClick = { pickAudio.launch(arrayOf("audio/*", "video/mp4")) },
                                        modifier = Modifier.fillMaxWidth(),
                                        enabled = !busy
                                    ) {
                                        Icon(Icons.Default.AudioFile, contentDescription = null)
                                        Spacer(Modifier.width(8.dp))
                                        Text("Wybierz MP3 / MP4 / WAV")
                                    }
                                    Text(selectedAudioName, style = MaterialTheme.typography.bodySmall)
                                    Button(
                                        onClick = {
                                            val uri = selectedAudio
                                            if (uri != null) scope.launch {
                                                busy = true
                                                progress = 0f
                                                status = "Dekoduję i analizuję próbkę…"
                                                try {
                                                    analysis = withContext(Dispatchers.IO) {
                                                        analyzer.analyze(uri) { value -> progress = value }
                                                    }
                                                    status = "Analiza zakończona. Próbka została przetworzona lokalnie."
                                                } catch (e: Exception) {
                                                    analysis = null
                                                    status = "Błąd analizy: ${e.message ?: e::class.java.simpleName}"
                                                } finally {
                                                    busy = false
                                                }
                                            }
                                        },
                                        enabled = selectedAudio != null && !busy,
                                        modifier = Modifier.fillMaxWidth()
                                    ) { Text(if (busy) "Analizuję…" else "Analizuj nagranie") }

                                    analysis?.let { result ->
                                        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                                            Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                                                Text("Wynik analizy", style = MaterialTheme.typography.titleMedium)
                                                Text("Czas: ${formatSeconds(result.durationSeconds)} • ${result.sampleRate} Hz • źródłowo ${result.sourceChannels} kan.")
                                                Text("RMS: ${formatNumber(result.rmsDbfs)} dBFS • szczyt: ${formatNumber(result.peak)}")
                                                Text("Próbki obcięte: ${formatNumber(result.clippingPercent)}% • cisza: ${formatNumber(result.silencePercent)}%")
                                                Text("Zdekodowano ${result.monoSamples} próbek mono do WAV PCM16.", style = MaterialTheme.typography.bodySmall)
                                            }
                                        }
                                        OutlinedTextField(
                                            value = profileName,
                                            onValueChange = { profileName = it },
                                            label = { Text("Nazwa profilu głosu") },
                                            modifier = Modifier.fillMaxWidth(),
                                            singleLine = true
                                        )
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Checkbox(
                                                checked = consentConfirmed,
                                                onCheckedChange = { consentConfirmed = it }
                                            )
                                            Text(
                                                "To mój głos albo mam wyraźną zgodę właściciela nagrania.",
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                        }
                                        Button(
                                            onClick = {
                                                scope.launch {
                                                    busy = true
                                                    try {
                                                        withContext(Dispatchers.IO) {
                                                            profileStore.save(
                                                                profileName,
                                                                result,
                                                                language,
                                                                consentConfirmed
                                                            )
                                                        }
                                                        profiles = withContext(Dispatchers.IO) { profileStore.list() }
                                                        consentConfirmed = false
                                                        status = "Zapisano lokalną próbkę głosu. Profil nie jest jeszcze podłączony do syntezy XTTS."
                                                    } catch (e: Exception) {
                                                        status = "Błąd zapisu profilu: ${e.message ?: e::class.java.simpleName}"
                                                    } finally {
                                                        busy = false
                                                    }
                                                }
                                            },
                                            enabled = !busy && consentConfirmed && profileName.trim().length in 2..40,
                                            modifier = Modifier.fillMaxWidth()
                                        ) { Text("Zapisz próbkę jako profil") }
                                    }
                                    Text(
                                        "Import i analiza nie klonują głosu. Właściwy pipeline XTTS ONNX wymaga jeszcze natywnej integracji i testu ARM64.",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
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
                            "Profile zawierają lokalne nagrania referencyjne i analizę. Nie są jeszcze używane przez silnik XTTS.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Screen.Models -> {
                        Text("Menedżer modeli", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text("Supertonic 3 INT8", style = MaterialTheme.typography.titleMedium)
                                Text(if (models.isSupertonicInstalled()) "Pobrany i zweryfikowany w pamięci aplikacji" else "Niepobrany — pobierze się przy pierwszej syntezie")
                                Button(
                                    onClick = {
                                        scope.launch {
                                            busy = true
                                            progress = 0f
                                            status = "Pobieram i weryfikuję Supertonic 3…"
                                            try {
                                                withContext(Dispatchers.IO) {
                                                    models.ensureSupertonic { value -> progress = value }
                                                }
                                                status = "Model Supertonic 3 jest gotowy"
                                            } catch (e: Exception) {
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
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("XTTS-v2 ONNX", style = MaterialTheme.typography.titleMedium)
                                Text("Silnik nie jest jeszcze zintegrowany. Nie pobieramy jego dużych plików, dopóki natywny pipeline nie będzie gotowy do ich użycia.")
                            }
                        }
                        if (busy) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                        StatusCard(status)
                    }

                    Screen.Settings -> {
                        Text("Ustawienia", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                Text("Przetwarzanie", style = MaterialTheme.typography.titleMedium)
                                Text("Silnik: CPU • ARM64 • model w prywatnym katalogu aplikacji")
                                Text(ramStats)
                                if (ramWarning) Text(
                                    "Mało dostępnej pamięci RAM. Zamknij zbędne aplikacje.",
                                    color = MaterialTheme.colorScheme.error
                                )
                                Text("Po pobraniu model Supertonic generuje offline.")
                                Text("NPU nie jest obecnie obsługiwane przez tę integrację.")
                                Text("Do syntezy Supertonic nie są wymagane Termux, Ubuntu PRoot, Python ani lokalny serwer.")
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
