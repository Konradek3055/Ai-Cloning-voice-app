package pl.kinotts.studio

import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.kinotts.studio.engine.SupertonicEngine
import pl.kinotts.studio.models.ModelManager
import java.io.File

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
    val supertonic = remember { SupertonicEngine() }

    var screen by remember { mutableStateOf(Screen.Studio) }
    var engine by remember { mutableStateOf("Supertonic 3") }
    var text by remember { mutableStateOf("Cześć! To jest test syntezy mowy w KinoTTS Studio.") }
    var selectedAudio by remember { mutableStateOf<Uri?>(null) }
    var selectedAudioName by remember { mutableStateOf("Nie wybrano pliku") }
    var status by remember { mutableStateOf("Gotowy") }
    var busy by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
    var sid by remember { mutableIntStateOf(0) }
    var speed by remember { mutableFloatStateOf(1f) }
    var steps by remember { mutableIntStateOf(8) }
    var language by remember { mutableStateOf("pl") }
    var output by remember { mutableStateOf<File?>(null) }
    var player by remember { mutableStateOf<MediaPlayer?>(null) }

    val pickAudio = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            try {
                context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: Exception) { }
            selectedAudio = uri
            selectedAudioName = uri.lastPathSegment ?: "Wybrany plik audio/wideo"
            status = "Plik wybrany. Analiza i klonowanie XTTS zostaną podłączone w natywnym silniku ONNX."
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
                            Text("Jedna aplikacja • lokalne modele", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                modifier = Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(16.dp),
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
                                    Text("XTTS-v2 będzie uruchamiany wewnątrz APK przez ONNX Runtime. W tej paczce projektowej port pipeline'u ONNX wymaga dokończenia i testu ARM64 — nie uruchamia Termuxa ani serwera.", color = MaterialTheme.colorScheme.error)
                                    OutlinedButton(onClick = { pickAudio.launch(arrayOf("audio/*", "video/mp4")) }, Modifier.fillMaxWidth()) {
                                        Icon(Icons.Default.AudioFile, null)
                                        Spacer(Modifier.width(8.dp))
                                        Text("Wybierz MP3 / MP4 / WAV")
                                    }
                                    Text(selectedAudioName, style = MaterialTheme.typography.bodySmall)
                                    Text("Po integracji silnik będzie analizował próbkę, zapisywał profile głosowe i klonował wyłącznie głos używany za zgodą właściciela.", style = MaterialTheme.typography.bodySmall)
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
                                        OutlinedTextField(value = language, onValueChange = {}, readOnly = true,
                                            modifier = Modifier.menuAnchor().fillMaxWidth(), label = { Text("Kod języka") })
                                        ExposedDropdownMenu(expanded = languageExpanded, onDismissRequest = { languageExpanded = false }) {
                                            listOf("pl", "en", "de", "fr", "es", "it", "cs", "uk").forEach { lang ->
                                                DropdownMenuItem(text = { Text(lang) }, onClick = { language = lang; languageExpanded = false })
                                            }
                                        }
                                    }
                                    Text("Tempo: ${"%.2f".format(speed)}×")
                                    Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.7f..1.4f)
                                    Text("Jakość / kroki: $steps")
                                    Slider(value = steps.toFloat(), onValueChange = { steps = it.toInt() }, valueRange = 4f..16f, steps = 11)
                                }
                            }
                        }
                        Button(
                            onClick = {
                                if (engine == "XTTS-v2") {
                                    status = "Silnik XTTS ONNX czeka na port i test w projekcie."
                                } else {
                                    scope.launch {
                                        busy = true
                                        status = "Sprawdzam model Supertonic 3…"
                                        try {
                                            val modelDir = withContext(Dispatchers.IO) {
                                                models.ensureSupertonic { p -> progress = p }
                                            }
                                            status = "Ładuję Supertonic 3…"
                                            val wav = withContext(Dispatchers.IO) {
                                                supertonic.load(modelDir)
                                                supertonic.generate(text, sid, speed, steps, language)
                                            }
                                            output = wav
                                            status = "Gotowe: ${wav.name}"
                                        } catch (e: Exception) {
                                            status = "Błąd: ${e.message ?: e::class.java.simpleName}"
                                        } finally { busy = false }
                                    }
                                }
                            },
                            enabled = !busy && (engine == "XTTS-v2" || text.isNotBlank()),
                            modifier = Modifier.fillMaxWidth()
                        ) { Text(if (busy) "Generuję…" else "Generuj mowę") }
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
                                            player = MediaPlayer().apply { setDataSource(file.absolutePath); prepare(); start() }
                                        }) { Text("Odtwórz") }
                                        OutlinedButton(onClick = { exportAudio.launch(file.name) }) {
                                            Text("Zapisz WAV w pamięci telefonu")
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Screen.Voices -> {
                        Text("Biblioteka głosów", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("Supertonic 3", style = MaterialTheme.typography.titleMedium)
                            Text("10 gotowych stylów głosu (ID 0–9). Wybierz głos w Studio.")
                            Text("Profile XTTS-v2 będą zapisywane lokalnie po ukończeniu natywnego silnika ONNX.", style = MaterialTheme.typography.bodySmall)
                        } }
                    }
                    Screen.Models -> {
                        Text("Menedżer modeli", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            Text("Supertonic 3 INT8", style = MaterialTheme.typography.titleMedium)
                            Text(if (models.isSupertonicInstalled()) "Pobrany do pamięci aplikacji" else "Niepobrany — pobierze się przy pierwszej syntezie")
                            Button(onClick = {
                                scope.launch {
                                    busy = true; status = "Pobieram Supertonic 3…"
                                    try { withContext(Dispatchers.IO) { models.ensureSupertonic { p -> progress = p } }; status = "Supertonic 3 gotowy" }
                                    catch (e: Exception) { status = "Błąd pobierania: ${e.message}" }
                                    finally { busy = false }
                                }
                            }, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("Pobierz / sprawdź model") }
                        } }
                        Card { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("XTTS-v2 ONNX INT8", style = MaterialTheme.typography.titleMedium)
                            Text("Model wymaga zatwierdzenia licencji na Hugging Face i natywnego portu pipeline'u do Androida. Pliki modelu będą pobierane do prywatnego katalogu tej aplikacji, a nie do Termuxa.")
                            Text("Postęp: wymaga integracji przez build agenta", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                        } }
                        if (busy) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                        StatusCard(status)
                    }
                    Screen.Settings -> {
                        Text("Ustawienia", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Card { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            Text("Przetwarzanie", style = MaterialTheme.typography.titleMedium)
                            Text("Tryb CPU • ARM64 • modele w prywatnym katalogu aplikacji")
                            Text("Modele są pobierane raz. Po pobraniu generowanie ma działać offline.")
                            Text("Nie są wymagane: Termux, Ubuntu PRoot, Python ani lokalny serwer.")
                        } }
                        Card { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("Prywatność i zgoda", style = MaterialTheme.typography.titleMedium)
                            Text("Klonuj wyłącznie własny głos albo głos osoby, która wyraźnie się zgodziła. Nagrania i profile pozostają w prywatnych danych aplikacji.")
                        } }
                    }
                }
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
