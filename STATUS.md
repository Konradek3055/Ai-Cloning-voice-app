# STATUS PROJEKTU — PRZECZYTAJ PRZED PRACĄ

Ten katalog jest natywnym projektem Android (Kotlin/Compose), a nie WebView ani serwerem. Nie zawiera Pythona, Termuxa ani Ubuntu.

Stan na moment przekazania:
- [x] Natywny szkielet aplikacji Android i cztery zakładki UI.
- [x] Zależność sherpa-onnx Android/Kotlin 2.0.1.
- [x] Pobieranie archiwum Supertonic 3 INT8 do katalogu prywatnego aplikacji.
- [x] Wywołanie natywnego API Supertonic z Kotlin.
- [x] Zapis wygenerowanego WAV i systemowy eksport przez Storage Access Framework.
- [x] Zależność ONNX Runtime Android.
- [ ] Pełny pipeline XTTS-v2 ONNX w Kotlin (krytyczny brak).
- [ ] Pobieranie wszystkich plików modelu XTTS i bezpieczny token HF (jeśli wymagany).
- [ ] Pełna analiza próbki i profilowanie głosu XTTS.
- [ ] Kompilacja APK oraz testy.
- [ ] Test rzeczywistej generacji na fizycznym telefonie ARM64.

Nie przedstawiaj projektu jako ukończonej aplikacji. Zadanie dla agenta znajduje się w `CLAUDE_TASK.md`, a gotowe polecenie startowe w `CLAUDE_PROMPT.txt`.
