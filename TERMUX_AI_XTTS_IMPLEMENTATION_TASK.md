# KinoTTS — zadanie implementacyjne

Pracuj bezpośrednio na tym repozytorium. Nie twórz atrap ani przycisków,
które udają działającą syntezę. Nie usuwaj działającego Supertonic.

## A. Trwałe dane
- Model i profile mają być poza katalogiem prywatnym aplikacji.
- Docelowo: Documents/KinoTTS/Models/XTTSv2
- Profile: Documents/KinoTTS/Voices/profiles.json
- Wyniki: Documents/KinoTTS/Output
- Bez folder pickera i bez ręcznego wpisywania ścieżek.
- Użyj MediaStore Downloads z RELATIVE_PATH dla plików modelu.
- Automatycznie twórz katalogi/znacznik przy pierwszym uruchomieniu.
- Przy starcie skanuj istniejące pliki i weryfikuj kompletność modelu.
- Przenieś istniejące pliki z filesDir/models/xttsv2 do trwałego magazynu
  przed usunięciem źródła. Nigdy nie kasuj źródła, dopóki kopia nie przejdzie
  kontroli rozmiaru i SHA-256.
- ORT może potrzebować zwykłych ścieżek plików: kopiuj z MediaStore do cache
  aplikacji i odtwarzaj cache z trwałego magazynu.
- Usuwanie modelu wyłącznie po jawnej akcji użytkownika i potwierdzeniu.
- Nie używaj getExternalFilesDir jako magazynu trwałego — znika po odinstalowaniu.

## B. Prawdziwa synteza XTTS-v2
Przeczytaj źródłowy pipeline XTTSv2-Streaming-ONNX oraz tokenizer.
Zaimplementuj kompletny przepływ:
audio referencyjne -> preprocessing/resampling -> conditioning encoder ->
speaker encoder -> tokenizer BPE i polskie czyszczenie tekstu ->
GPT autoregresyjny z KV cache -> HiFi-GAN vocoder -> WAV 24 kHz.
Obsłuż język polski (pl), anulowanie, postęp, błędy i brak RAM.
Nie zastępuj pipeline'u samym otwieraniem czterech sesji ONNX.
Nie wymyślaj nazw tensorów ani kształtów — odczytaj je z grafów i kodu
referencyjnego. Zaimplementuj dokładnie wymagane wejścia/wyjścia.
Wykorzystaj istniejący XtTsOnnxRuntime i XttsModelManager zamiast dublować kod.

## C. Profile głosowe
- Import próbki WAV/M4A/MP3 przez systemowy picker pliku.
- Profil zapisuje próbkę i metadane w trwałym Documents/KinoTTS/Voices.
- Lista profili musi działać po reinstalacji.
- Synteza używa wybranego profilu.
- Klonowanie wyłącznie własnego głosu albo nagrania z wyraźną zgodą.
- Zachowaj licencję modelu CC-BY-NC-4.0.

## D. UI i jakość
- Dodaj osobny ekran XTTS: wybór profilu, tekst, język, generowanie,
  postęp, anulowanie, odtwarzanie i eksport WAV.
- Zachowaj ekran Supertonic i dotychczasowe funkcje.
- Dodaj czytelny status modelu: brak / pobieranie / gotowy / uszkodzony.
- Dodaj przycisk Usuń model z potwierdzeniem.
- Ikona aplikacji w mipmapach, spójna z nazwą KinoTTS Studio.
- Zaktualizuj README i THIRD_PARTY_NOTICES.

## E. Testy obowiązkowe
- Gradle assembleDebug.
- Test kompilacji ARM64.
- Test otwarcia wszystkich sesji ONNX.
- Test rzeczywistej syntezy na krótkim polskim tekście z własną próbką.
- Test skanowania modelu i profili po ponownym uruchomieniu.
- Test migracji prywatnego modelu bez kasowania oryginału przed weryfikacją.
- Nie deklaruj XTTS jako gotowego, jeśli test generowania WAV nie przejdzie.
- Nie zmieniaj applicationId.
- Nie usuwaj plików użytkownika ani modelu podczas aktualizacji.
