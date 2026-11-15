plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val ciKeystorePath = System.getenv("KINOTTS_KEYSTORE_PATH")
val ciStorePassword = System.getenv("KINOTTS_STORE_PASSWORD")
val ciKeyAlias = System.getenv("KINOTTS_KEY_ALIAS")
val ciKeyPassword = System.getenv("KINOTTS_KEY_PASSWORD")

val useCiSigning =
    !ciKeystorePath.isNullOrBlank() &&
    !ciStorePassword.isNullOrBlank() &&
    !ciKeyAlias.isNullOrBlank() &&
    !ciKeyPassword.isNullOrBlank()

android {
    namespace = "pl.kinotts.studio"
    compileSdk = 35

    defaultConfig {
        applicationId = "pl.kinotts.studio"
        minSdk = 28
        targetSdk = 35
        versionCode = 3
        versionName = "2.1.0-standalone"
        ndk { abiFilters += setOf("arm64-v8a") }
    }

    signingConfigs {
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

    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        jniLibs {
            useLegacyPackaging = false
            // Sherpa 2.0.1 also contains libonnxruntime.so (1.28.2).
            // The ORT 1.32.0 dependency is declared first and is checked at runtime.
            pickFirsts += "**/libonnxruntime.so"
        }
        resources.excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1")
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.05.00"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    // Keep this first: it provides the Java API/JNI and the single ORT 1.32.0 runtime.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")

    // Official sherpa-onnx Android AAR (Kotlin API + arm64 native runtime).
    implementation("com.github.k2-fsa:sherpa-onnx:2.0.1") {
        exclude(module = "sherpa-onnx-jvm")
    }

    // Used by the native XTTS-v2 ONNX port. No Python server is part of this app.


    implementation("org.apache.commons:commons-compress:1.28.0")
}
