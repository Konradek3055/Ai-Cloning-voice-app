# ZADANIE DLA CLAUDE — DOKOŃCZ PRAWDZIWE KINOTTS STUDIO (JEDNO APK)

## Cel — nie zmieniaj go
Zbuduj **jedną samodzielną aplikację Android APK** dla ARM64. Użytkownik instaluje tylko `KinoTTS Studio`. Nie ma żadnego drugiego programu ani komponentu, który użytkownik musi osobno uruchamiać.

**ZABRONIONA ARCHITEKTURA:** Termux, Ubuntu PRoot, Chaquopy serwer, Python HTTP server, localhost, WebView jako panel do zewnętrznego backendu, osobny proces/usługa wymagająca ręcznego startu. Nie twórz żadnego serwera.

Wszystkie funkcje i silniki muszą być uruchamiane bezpośrednio przez kod w procesie APK (Kotlin/JNI/ONNX Runtime). Modele pobieraj z internetu do prywatnego katalogu aplikacji (`context.noBackupFilesDir` albo `context.filesDir`). Po pobraniu aplikacja ma działać offline.

## Najważniejsze wymagania funkcjonalne
1. Native Kotlin + Jetpack Compose, pełny ekran Studio, Głosy, Modele, Ustawienia.
2. Wczytywanie MP3, MP4, WAV przez Storage Access Framework. Dekodowanie audio wewnątrz Androida (MediaExtractor/MediaCodec lub biblioteka natywna), mono PCM, normalizacja i resampling.
3. Analiza próbki: długość, sample rate, kanały, głośność/RMS, clipping, cisza i segmenty mowy/VAD. Wyniki pokaż jako ocenę jakości próbki; nie twierdź, że wykryto tożsamość osoby ani że jest to dowód zgody.
4. XTTS-v2: pełny zero-shot voice cloning, język polski `pl`, profile głosu zapisywane lokalnie (referencyjne audio i/lub wyliczone conditioning latents + speaker embedding). Użytkownik wpisuje tekst, wybiera profil, reguluje temperaturę, top-k, top-p, repetition penalty, prędkość, generuje WAV, odsłuchuje i eksportuje plik.
5. Supertonic 3: 10 gotowych głosów (`sid 0..9`), język `pl` i inne obsługiwane języki, tempo i `numSteps`, zapis WAV/MP3.
6. Aplikacja sama pobiera modele na żądanie; pasek postępu, anulowanie, wznowienie/bezpieczny restart, SHA-256/checksum, kontrola wolnego miejsca, naprawa niepełnego pobrania.
7. Nie ładuj obu ciężkich modeli do RAM jednocześnie. Zwolnij silnik przy przełączaniu. Praca CPU na ARM64, bez wymogu CUDA.
8. Brak telemetrii i brak wysyłania referencyjnych nagrań na serwer. Niech wszystko zostanie w pamięci aplikacji.

## Supertonic 3 — istnieje natywne API
W projekcie jest już `SupertonicEngine.kt` i pobieranie oficjalnego archiwum modelu INT8. Użyj oficjalnej biblioteki `sherpa-onnx` Android/Kotlin 2.0.1.
- Kotlin API: https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/kotlin-api/Tts.kt
- Przykład Supertonic Kotlin: https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_supertonic_tts.kt
- Dokumentacja modelu: https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
- Archiwum modelu INT8: https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
Wersja 3 obsługuje 31 języków, w tym polski. API generowania używa `OfflineTts`, `OfflineTtsSupertonicModelConfig` i `GenerationConfig(extra = mapOf("lang" to "pl"))`.

## XTTS-v2 — obowiązkowa implementacja, nie placeholder
Nie używaj `coqui-tts` PyTorch/Python z Termuxa. Przenieś pipeline do Kotlin/Java i wykonuj ONNX w procesie Androida.

Punkt odniesienia dla kodu pipeline'u (Apache-2.0):
- Repo: https://github.com/sjamthe/XTTSv2-Streaming-ONNX
- Pipeline: https://github.com/sjamthe/XTTSv2-Streaming-ONNX/blob/main/xtts_streaming_pipeline.py
- Orchestrator: https://github.com/sjamthe/XTTSv2-Streaming-ONNX/blob/main/xtts_onnx_orchestrator.py
- Tokenizer: https://github.com/sjamthe/XTTSv2-Streaming-ONNX/blob/main/xtts_tokenizer.py
Repo opisuje pełny pipeline ONNX z czterema elementami: conditioning encoder, speaker encoder, autoregresyjny GPT-2 z KV cache i HiFi-GAN vocoder. Jest też wariant `gpt_model_int8.onnx`.

Model pack:
- Źródło INT8: https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX/tree/main/xtts_onnx
- To repo jest gated: użytkownik musi sam zalogować się do Hugging Face i zaakceptować warunki CC BY-NC 4.0. Dodaj w aplikacji ekran z informacją i polem na HF read token, jeśli pobranie tego modelu go wymaga. Token zapisuj wyłącznie zaszyfrowany przez Android Keystore; nigdy nie loguj go i nie wysyłaj nigdzie poza Hugging Face.
- W katalogu modelu są m.in. `conditioning_encoder.onnx`, `speaker_encoder.onnx`, `gpt_model_int8.onnx`, `hifigan_vocoder.onnx`, `mel_stats.npy`, `metadata.json`, `vocab.json` i `embeddings/*`.
- Jeśli użytkownik nie może uzyskać dostępu do gated pack, alternatywnie obsłuż publiczny export: https://huggingface.co/AMD-PAVS-AI/XTTS-v2/tree/main — ale pobieraj tylko potrzebne pliki, nie cały 4.92GB repo; tam dekoder GPT jest FP32 i może mieć większe zapotrzebowanie na RAM.

Port XTTS musi obejmować:
- BPE tokenizację kompatybilną z vocab/model export.
- MediaExtractor/MediaCodec decode audio z MP3/MP4/WAV.
- resampling do 22050 Hz i 16000 Hz;
- STFT/mel spectrogram: n_fft=2048, hop=256, win=1024, 80 mels, 0..8000Hz, Slaney filters, log clamp 1e-5, mel normalization z `mel_stats.npy`; referencja max 6 s jak w pipeline.
- conditioning encoder + speaker encoder; zapis wynikowych latents i embedding.
- GPT ONNX autoregressive loop z KV-cache, stop token, sampling (temperature/top-k/top-p/repetition penalty), language token.
- HiFi-GAN vocoder, stream/crossfade, output 24kHz.
- Użyj `com.microsoft.onnxruntime:onnxruntime-android` (aktualnie 1.31.1) — `OrtEnvironment`, `OrtSession`, `OnnxTensor`. Nie uruchamiaj Pythona.

**BARDZO WAŻNE:** nie wystarczy pobrać model files i pokazać przycisku. Nie oznaczaj XTTS jako działającego dopóki nie zaimplementujesz pełnego pipeline'u i nie przejdziesz testu generowania na rzeczywistym urządzeniu ARM64. Jeśli tego nie potrafisz dokończyć, napisz wyraźnie, które pliki/kroki pozostały, ale nie zastępuj XTTS atrapą ani innym modelem.

## Wymagania pamięci / telefonu
Użytkownik ma Samsung Galaxy S25 FE, ARM64, Android 17, około 7 GB RAM dostępnej w środowisku PRoot (stan może się zmieniać) i ostatnio 23 GB wolnego miejsca. Zastosuj bezpieczne limity:
- 2–4 wątki CPU, domyślnie 2 dla XTTS;
- ogranicz batch/chunk długość, użyj streaming i `OrtSession` options z pamięcią;
- nie twórz wielu kopii tensorów i nie trzymaj jednocześnie obu silników;
- przed pobraniem sprawdź wolne miejsce; przed generowaniem ostrzeż o pamięci;
- przechwyć OOM, wyświetl zrozumiały komunikat zamiast crasha.

## Rozwój i wydanie
- Popraw każdy compile error i warning dotyczący API.
- `./gradlew :app:assembleDebug` musi zakończyć się kodem 0.
- Uruchom testy jednostkowe i instrumentacyjne, jeśli środowisko pozwala.
- Otwórz APK i sprawdź `aapt dump badging`/APK Analyzer: `pl.kinotts.studio`, `arm64-v8a`, minSdk 28.
- Przeprowadź próbę na prawdziwym Androidzie ARM64: pobierz Supertonic, wygeneruj i zapisz WAV; następnie pobierz XTTS i wykonaj test krótkiego polskiego tekstu z referencyjnym WAV.
- Dostarcz gotowy `.apk` jako artefakt. Dołącz źródła i raport testów.
- Nie deklaruj testu na telefonie, jeśli go fizycznie nie wykonałeś.

## Warunki akceptacji — wszystkie wymagane
- [ ] Użytkownik instaluje tylko jeden plik APK.
- [ ] Po usunięciu Termuxa/Ubuntu aplikacja nadal działa.
- [ ] Nie ma localhost HTTP server, WebView panelu ani oddzielnego backendu.
- [ ] Aplikacja pobiera modele do własnej pamięci.
- [ ] Supertonic 3 generuje mowę po polsku i zapisuje WAV.
- [ ] XTTS-v2 klonuje głos z MP3/MP4/WAV i generuje mowę po polsku.
- [ ] Po pobraniu modeli główne funkcje działają bez internetu.
- [ ] Aplikacja prawidłowo komunikuje wymagania licencji.
