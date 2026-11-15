from pathlib import Path
from datetime import datetime
import shutil
import sys

ROOT = Path.cwd()
GRADLE = ROOT / "app/build.gradle.kts"
WORKFLOW = ROOT / ".github/workflows/build-apk.yml"
GITIGNORE = ROOT / ".gitignore"

def backup(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(path.name + ".bak-" + stamp)
    shutil.copy2(path, dst)
    print("Backup:", dst)

for p in (GRADLE, WORKFLOW):
    if not p.is_file():
        raise SystemExit("BRAK PLIKU: " + str(p))

g = GRADLE.read_text()
w = WORKFLOW.read_text()

# ---------- Gradle: obsługa stałego klucza z GitHub Secrets ----------
if "KINOTTS_KEYSTORE_PATH" not in g:
    anchor = "android {\n"
    if anchor not in g:
        raise SystemExit("Nie znaleziono bloku android {} w Gradle.")

    signing_vars = '''val ciKeystorePath = System.getenv("KINOTTS_KEYSTORE_PATH")
val ciStorePassword = System.getenv("KINOTTS_STORE_PASSWORD")
val ciKeyAlias = System.getenv("KINOTTS_KEY_ALIAS")
val ciKeyPassword = System.getenv("KINOTTS_KEY_PASSWORD")

val useCiSigning =
    !ciKeystorePath.isNullOrBlank() &&
    !ciStorePassword.isNullOrBlank() &&
    !ciKeyAlias.isNullOrBlank() &&
    !ciKeyPassword.isNullOrBlank()

'''

    g = g.replace(anchor, signing_vars + anchor, 1)

    anchor = "    buildFeatures { compose = true }"
    signing_block = '''    signingConfigs {
        if (useCiSigning) {
            create("ci") {
                storeFile = file(ciKeystorePath!!)
                storePassword = ciStorePassword
                keyAlias = ciKeyAlias
                keyPassword = ciKeyPassword
            }
        }
    }

    buildTypes {
        getByName("debug") {
            if (useCiSigning) {
                signingConfig = signingConfigs.getByName("ci")
            }
        }
    }

'''
    if anchor not in g:
        raise SystemExit("Nie znaleziono buildFeatures w Gradle.")
    g = g.replace(anchor, signing_block + anchor, 1)

# ---------- Workflow: wczytanie klucza z GitHub Secrets ----------
if "Configure stable APK signing" not in w:
    anchor = "      - name: Build debug APK\n"
    setup = '''      - name: Configure stable APK signing
        env:
          KINOTTS_KEYSTORE_BASE64: ${{ secrets.KINOTTS_KEYSTORE_BASE64 }}
          KINOTTS_STORE_PASSWORD: ${{ secrets.KINOTTS_STORE_PASSWORD }}
          KINOTTS_KEY_ALIAS: ${{ secrets.KINOTTS_KEY_ALIAS }}
          KINOTTS_KEY_PASSWORD: ${{ secrets.KINOTTS_KEY_PASSWORD }}
        run: |
          set -euo pipefail
          if [[ -n "$KINOTTS_KEYSTORE_BASE64" ]]; then
            KEYSTORE="$RUNNER_TEMP/kinotts-signing.jks"
            printf '%s' "$KINOTTS_KEYSTORE_BASE64" | base64 --decode > "$KEYSTORE"
            test -s "$KEYSTORE"
            {
              echo "KINOTTS_KEYSTORE_PATH=$KEYSTORE"
              echo "KINOTTS_STORE_PASSWORD=$KINOTTS_STORE_PASSWORD"
              echo "KINOTTS_KEY_ALIAS=$KINOTTS_KEY_ALIAS"
              echo "KINOTTS_KEY_PASSWORD=$KINOTTS_KEY_PASSWORD"
            } >> "$GITHUB_ENV"
            echo "Używam stałego klucza podpisującego."
          else
            echo "::warning::Brak KINOTTS_KEYSTORE_BASE64 — używam tymczasowego podpisu debug."
          fi

'''
    if anchor not in w:
        raise SystemExit("Nie znaleziono kroku Build debug APK.")
    w = w.replace(anchor, setup + anchor, 1)

# ---------- Usunięcie błędnego porównania losowego certyfikatu ----------
start = w.find("      - name: Verify stable update signature and version")
if start >= 0:
    end = w.find("      - name: Upload APK artifact", start)
    if end < 0:
        raise SystemExit("Nie znaleziono kroku Upload APK artifact.")
    replacement = '''      - name: Verify APK signature
        run: |
          set -euo pipefail
          APK="$(find app/build/outputs/apk/debug -maxdepth 1 -type f -name '*.apk' -print -quit)"
          test -n "$APK"
          APKSIGNER="$(find "$ANDROID_HOME/build-tools" -type f -name apksigner | sort -V | tail -n 1)"
          test -x "$APKSIGNER"
          "$APKSIGNER" verify --verbose --print-certs "$APK" | tee apk-signing.txt
          grep -Eq "Signer #1 certificate SHA-256 digest: [0-9a-fA-F]{64}" apk-signing.txt
          if [[ -n "${KINOTTS_KEYSTORE_PATH:-}" ]]; then
            echo "APK podpisany stałym kluczem KinoTTS."
          else
            echo "::warning::Podpis debug jest tymczasowy. Skonfiguruj GitHub Secrets, aby aktualizacje miały stały podpis."
          fi

'''
    w = w[:start] + replacement + w[end:]
else:
    raise SystemExit("Nie znaleziono starego kroku weryfikacji podpisu.")

# Diagnostyka podpisu w artefakcie
w = w.replace(
    "            apk-info.txt\n",
    "            apk-info.txt\n            apk-signing.txt\n"
)

backup(GRADLE)
backup(WORKFLOW)
GRADLE.write_text(g)
WORKFLOW.write_text(w)

# Klucze nigdy nie mogą trafić do repozytorium.
if GITIGNORE.exists():
    gi = GITIGNORE.read_text()
else:
    gi = ""
for line in ("/signing/", "*.jks", "*.keystore"):
    if line not in gi.splitlines():
        gi += ("\n" if gi and not gi.endswith("\n") else "") + line + "\n"
GITIGNORE.write_text(gi)

# Zadanie dla lokalnego agenta AI — właściwa implementacja XTTS.
task = r"""# KinoTTS — zadanie implementacyjne

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
"""

task_path = ROOT / "TERMUX_AI_XTTS_IMPLEMENTATION_TASK.md"
task_path.write_text(task, encoding="utf-8")

print("\nPATCH CI ZASTOSOWANY.")
print("Zmieniono:")
print(" - app/build.gradle.kts")
print(" - .github/workflows/build-apk.yml")
print(" - .gitignore")
print("Utworzono:", task_path)
print("\nUWAGA: To naprawia błąd kontroli podpisu i dodaje obsługę stałego klucza.")
print("Nie twierdzi, że pełny pipeline XTTS jest już zaimplementowany.")
