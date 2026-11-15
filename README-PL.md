# KinoTTS Studio — samodzielna aplikacja Android (ARM64)

To jest nowy projekt natywnej aplikacji Android. Nie jest to WebView do lokalnego serwera i nie wymaga Termux, Ubuntu PRoot, Pythona ani ręcznego uruchamiania backendu.

## Stan tego projektu

- **Aplikacja i ekran Android:** przygotowane w Kotlin + Jetpack Compose.
- **Pobieranie Supertonic 3 INT8 do prywatnych danych aplikacji:** przygotowane.
- **Generowanie Supertonic 3:** podłączone przez natywne API sherpa-onnx Kotlin.
- **XTTS-v2:** opisano dokładną ścieżkę natywnej integracji ONNX w `CLAUDE_TASK.md`, ale pełny port pipeline'u z referencyjnego Pythona do Kotlin/ONNX Runtime wymaga dokończenia. Nie udajemy, że ta funkcja już działa.
- **APK:** nie zostało zbudowane ani przetestowane w tym środowisku, ponieważ nie ma Android SDK/Gradle. Projekt przygotowano do dokończenia i zbudowania w środowisku Claude z SDK.

## Jak przekazać projekt Claude

1. Wyślij Claude cały ZIP.
2. Otwórz `CLAUDE_TASK.md` i skopiuj całą jego treść jako polecenie.
3. Wymagaj gotowego `.apk`, a nie samego kolejnego ZIP-a.
4. Claude ma dokończyć XTTS ONNX w kodzie Androida i potwierdzić rzeczywiste generowanie na ARM64. Nie akceptuj atrapy ani serwera Termux.

## Architektura docelowa

- Jedno APK `pl.kinotts.studio`.
- Kotlin/Compose jako UI.
- sherpa-onnx Android/Kotlin dla Supertonic 3.
- ONNX Runtime Android dla XTTS-v2 ONNX.
- Modele pobierane na żądanie i przechowywane w `filesDir` aplikacji.
- Po pobraniu modeli aplikacja działa offline.

## Build

Wymagania: JDK 17, Android SDK Platform 35, Gradle 8.11+.

```bash
./gradlew :app:assembleDebug
```

APK (po udanym buildzie):

`app/build/outputs/apk/debug/app-debug.apk`

## Licencje

- sherpa-onnx: Apache-2.0.
- ONNX Runtime: MIT.
- Referencyjny kod XTTS ONNX: Apache-2.0.
- Model XTTS ONNX w gated repo: CC BY-NC 4.0 — użytkownik musi zaakceptować warunki modelu przed pobraniem. Nie komercjalizować ani nie redystrybuować wag bez odpowiednich praw.
- Supertonic 3 ma odrębną licencję modelu Supertone; pokaż jej warunki przed pobraniem i nie usuwaj informacji licencyjnych.

Przeznaczony do pracy z własnym głosem albo nagraniem osoby, która wyraziła zgodę.


## Automatyczne budowanie APK przez GitHub Actions

Workflow `.github/workflows/build-apk.yml` uruchamia się po każdym `push` albo ręcznie przez `workflow_dispatch`. Instalowane są JDK 17, Gradle 8.11.1 oraz Android SDK Platform 35 i Build Tools 35.0.0. Build kończy się błędem, jeśli kompilacja, identyfikator aplikacji, `minSdk` lub ABI `arm64-v8a` nie przejdą kontroli.

Gotowy plik trafia do artefaktu `KinoTTS-Studio-debug-APK-arm64` w zakładce Actions danego uruchomienia. Diagnostyka trafia do osobnego artefaktu.
