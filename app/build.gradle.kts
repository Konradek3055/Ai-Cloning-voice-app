plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "pl.kinotts.studio"
    compileSdk = 35

    defaultConfig {
        applicationId = "pl.kinotts.studio"
        minSdk = 28
        targetSdk = 35
        versionCode = 2
        versionName = "2.0.0-standalone"
        ndk { abiFilters += setOf("arm64-v8a") }
    }

    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        jniLibs { useLegacyPackaging = false }
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

    // Official sherpa-onnx Android AAR (Kotlin API + arm64 native runtime).
    implementation("com.github.k2-fsa:sherpa-onnx:2.0.1") {
        exclude(module = "sherpa-onnx-jvm")
    }

    // Used by the native XTTS-v2 ONNX port. No Python server is part of this app.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")

    implementation("org.apache.commons:commons-compress:1.28.0")
}
