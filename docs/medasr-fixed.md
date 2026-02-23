# MedASR Sherpa ONNX Integration - Fixed & Optimized
This document details the successful resolution of the Sherpa ONNX integration issues for the MedGemma Forensic application.

## 1. The Core Issue & Failed Attempts
The original integration faced a severe version conflict between the **Sherpa ONNX AAR** (compiled with Kotlin 1.7.20) and the project's **Kotlin Compiler** (2.0.21).

Initial attempts to resolve this were **unsuccessful** and involved complex, unstable workarounds:
*   **AAR Metadata Stripping**: Attempted to manually remove `@kotlin.Metadata` annotations from the AAR byte-code to force "Java-mode" compatibility. This failed to produce a stable build.
*   **Compiler Version Experiments**: Attempted to force compatibility by adjusting Kotlin versions and flags, which did not resolve the fundamental ABI mismatch.
*   **"Hacked" Initialization**: Tried to instantiate classes using Java-reflection style workarounds, but this led to "Redeclaration" errors and unstable runtime behavior.

## 2. The Solution: Source Code Integration (Documentation-Led)
The breakthrough occurred when **you (the User) provided the official documentation and repository structure**. This revealed that the correct, intended way to resolve such deep incompatibilities is not to "hack" the binary, but to integrate the source directly.

Using the documentation you provided, we adopted a **Native Source Integration** strategy:

### How It Works
1.  **Adopted the Source**: Following your guidance, we downloaded the actual Kotlin source files (`OfflineRecognizer.kt`, etc.) from the repository.
2.  **Native Compilation**: By compiling these files within *your* specific project environment (Kotlin 2.0.21), we eliminated the version gap entirely.
3.  **Engine Preservation**: We kept only the necessary native `.so` (C++) libraries, discarding the problematic AAR wrapper.

## 3. Implementation Details

### A. File Structure Changes
We removed the `.aar` dependency and established a direct source structure.

**1. Kotlin Source Files**
Location: `app/src/main/kotlin/com/k2fsa/sherpa/onnx/`
Added files:
*   `OfflineRecognizer.kt` (The main entry point)
*   `OnlineRecognizer.kt` (If used)
*   `FeatureConfig.kt`
*   `OfflineStream.kt`
*   `OfflineModelConfig.kt`
*   `QnnConfig.kt`
*   `HomophoneReplacerConfig.kt`

**2. Native Libraries (JNI)**
Location: `app/src/main/jniLibs/`
Extracted architectures:
*   `arm64-v8a/` (Modern Android devices)
*   `armeabi-v7a/` (Older devices)
*   `x86_64/` (Emulators)

Libraries included:
*   `libsherpa-onnx-jni.so`
*   `libonnxruntime.so`
*   `libc++_shared.so`

### B. Gradle Configuration (`build.gradle.kts`)
*   **Removed**: `implementation(files("libs/sherpa-onnx-1.12.24.aar"))`
*   **Removed**: Any `fileTree` dependencies that might implicitly pick up AARs.
*   **Verified**: `abiFilters` in `android.defaultConfig` ensures only valid architectures are packaged in the APK.

### C. Code Refactoring (`MedASRManager.kt`)
The `MedASRManager` was refactored to use idiomatic Kotlin features, which is now possible since we are compiling from source:
*   **Named Arguments**: Used for clarity (e.g., `FeatureConfig(sampleRate = 16000)`).
*   **Coroutines**: Switched to `suspend` functions and `Dispatchers.IO` to prevent Main Thread blocking (ANR) during model loading.
*   **Error Handling**: added explicit `try-catch` blocks with logging for robust initialization.

## 4. How to Update Sherpa ONNX
Since this is a manual integration, updating the library in the future (e.g., to v1.13 or v2.0) follows this process:

1.  **Update Source Code**:
    *   Go to the [Sherpa ONNX GitHub Repository](https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnx/sherpa-onnx/src/main/java/com/k2fsa/sherpa/onnx).
    *   Download the updated `.kt` files.
    *   Replace the files in `app/src/main/kotlin/com/k2fsa/sherpa/onnx/`.

2.  **Update Native Libraries**:
    *   Download the new release `.aar` file.
    *   Unzip the `.aar` (using 7zip or renaming to .zip).
    *   Copy the contents of the `jni/` folder to `app/src/main/jniLibs/`, overwriting existing files.

3.  **Clean Build**:
    *   Run `./gradlew clean` to clear any cached artifacts.

## 5. Troubleshooting Guide

### Issue: "Redeclaration: class FeatureConfig"
*   **Cause**: You have both the Kotlin source file AND a binary (JAR/AAR) or Java file defining the same class.
*   **Fix**:
    1.  Delete any `.aar` files in `app/libs/`.
    2.  Check for and delete any Java sources in `app/src/main/java/com/k2fsa/`.
    3.  Run `./gradlew clean`.

### Issue: "UnsatisfiedLinkError"
*   **Cause**: The native library `.so` file was not found or is incompatible with the device architecture.
*   **Fix**:
    1.  Ensure `app/src/main/jniLibs/` contains the folder for your device's ABI (e.g., `arm64-v8a`).
    2.  Ensure `libsherpa-onnx-jni.so` is inside that folder.

### Issue: "UI Freeze / ANR on Startup"
*   **Cause**: Model loading (150MB+) is happening on the Main Thread.
*   **Fix**: Ensure `MedASRManager.init(context)` is called within `CoroutineScope(Dispatchers.IO).launch` or `withContext(Dispatchers.IO)`. We have already applied this fix in the current code.


### Links: 
*   [Sherpa ONNX GitHub Repository](https://github.com/k2-fsa/sherpa-onnx)

*   [Sherpa ONNX Build Instructions](https://k2-fsa.github.io/sherpa/onnx/android/build-sherpa-onnx.html)

*   [Sherpa ONNX MedASR Model](https://huggingface.co/csukuangfj/sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tree/main) ==we are using this model==

*   [Sherpa ONNX ASR Models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)