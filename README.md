
# 🕵️ MedGemma FORENSIC: The Autonomous Pathologist for the Missing Billion

> *Deploying Datacenter-Class Medical Intelligence to the Offline Edge.*

![Project Badge](https://img.shields.io/badge/Status-Prototype_Complete-success)![Platform](https://img.shields.io/badge/Platform-Android_Edge-green)![AI Models](https://img.shields.io/badge/AI-Gemma_3_1B_%7C_MedSigLIP_%7C_MedASR-blue)

### 🔗 Official Submission Links
*   **🎥 [Watch the 3-Minute Demo Video Here](https://drive.google.com/file/d/1OCYVyPv4xiclwgFENqZe1GKJV6mjcd-1/view?usp=drive_link)**
*   **🕹️ [Try the Live Gradio Web Demo](https://huggingface.co/spaces/smfaisal/medgemma-forensic)**
*   **💻 [Full Android & Edge Architecture Source Code (GitHub)](https://github.com/smjahid012/medgemma-forensic)**

---

## 🌍 The Mission
In the "Last Mile" of healthcare—conflict zones, remote villages, and disaster sites—the internet is a luxury that doctors cannot afford. Disease moves faster than data.

**MedGemma FORENSIC** is a **Self-Contained Artificial Intelligence System** engineered to operate in invalid zero-connectivity environments. I move intelligence from the Cloud to the Edge, empowering frontline workers to detect invisible outbreaks before they become pandemics.

---

## 🚀 The Engineering Breakthrough: "The Impossible Stack"

Running a single AI model on a phone is standard. Running a **Multimodal Diagnostic Suite** (LLM + Vision + Audio + Anomaly Detection) on a consumer Android device ($200, 4GB RAM) was considered impossible.

**I solved the "4GB Memory Paradox".**

My architecture orchestrates **4.3GB of Quantized AI Models** into a 3.5GB RAM envelope using a custom-built **Sequential Lifecycle Manager ("The Traffic Cop")**.

| Component | Model / Tech | Role | Innovation |
| :--- | :--- | :--- | :--- |
| **🧠 The Brain** | **Gemma 3 1B** (LiteRT) | Clinical Reasoning | **Zero-Latency Design:** *I prioritize stability. While MedGemma 1.5 4B is supported, I default to Gemma 3 1B to avoid Decoder KV Cache memory spikes on low-end hardware.* |
| **👁️ The Eye** | **MedSigLIP** (Google Research) | Visual Forensics | **Custom TFLite Conversion:** *I manually converted Google's SigLIP to TFLite and implemented a `LOGIT_SCALE=100.0f` scalar in the Android pipeline to restore sensitivity for "necrotic" patterns.* |
| **👂 The Ear** | **MedASR** (Sherpa ONNX) | Verbal Autopsy | **Raw Audio Pipeline:** *Direct MediaCodec decoding (M4A/WAV -> PCM FloatArray) for studio-quality transcription in the field.* |
| **📉 Pattern Hunter** | **Probabilistic Engine** | Outbreak Detection | **Forensic Bridge:** *Maps 60+ visual findings to a 25-dim epidemiological vector.* |
| **🕸️ Nervous System** | **P2P Mesh Network** | Offline Sync | **Worker-Anchor Protocol:** *Secure, manual handshake to transfer `CaseFile` protobufs without internet.* |

```
medgemma-forensic/                        ← GitHub Repo Root
│
├── 📄 README.md                          ← Main project page
├── 📄 LICENSE                            ← Apache 2.0 text
│
├── 📁 docs/
│   ├── 📄 impact-report.md               ← The 3-page write-up
│   ├── 📄 Medgemma_Forensic.ipynb        ← Kaggle notebook
│   ├── 📄 medsiglip-fixed.md             ← MedSigLIP technical details
│   └── 📄 medasr-fixed.md                ← MedASR technical details
│
├── 📁 medgemma-ram-test/
│   ├── 📄 ram-report.md                   ← Memory profiling report
│   └── 🖼️ native-test-visual.png          ← Android Studio RAM screenshot
│
└── 📁 core-architecture-source/           ← Core AI module implementations
    ├── 📄 ModelLifecycleManager.kt        ← Proves the "Traffic Cop" RAM management
    ├── 📄 MedSigLIPManager.kt             ← Proves the Logit Scale & L2 Normalization math
    ├── 📄 MedASRManager.kt                ← Proves the raw audio pipeline engineering
    ├── 📄 SentencePieceTokenizer.kt       ← Proves tokenizer built from scratch
    ├── 📄 DiseaseToSymptomMapper.kt       ← Proves the 25-dimensional "Rosetta Stone"
    └── 📄 AnomalyEngine.kt                ← Proves the local Epidemic Pattern matching
```


## 🏗️ Architecture: The "Worker-Anchor" Mesh

I respect the hierarchy of field medicine. Not every phone needs to be a supercomputer.

### 1. 👷 The Worker Node (Field Medic)
*   **Role:** Usage-focused, Battery-conscious.
*   **Action:** Captures Evidence (Photo + Audio + Vitals).
*   **AI Task:** 
    *   Runs **MedSigLIP** to categorize the lesion.
    *   Runs **MedASR** to transcribe the patient history.
    *   **Does NOT** run heavy reasoning. It packages data into a `CaseFile.proto` (highly compressed semantic data).
*   **Transfer:** Initiates a **Secure Handshake** (Manual WiFi Direct/BLE) to offload data to the Anchor.

### 2. ⚓ The Anchor Node (Command Center)
*   **Role:** Analysis-focused, plugged-in (or high battery).
*   **Action:** Receives `CaseFile` streams from multiple Workers.
*   **AI Task:**
    *   **"The Traffic Cop":** Unloads Listeners -> Loads **Gemma 3 1B**.
    *   **Reasoning:** Synthesizes the Worker's evidence into a `ForensicReport`.
    *   **Anomaly Engine:** Cross-references the new case with local history to detect **Spatiotemporal Clusters** (e.g., *"3 cases of Necrosis in Village A within 48 hours"*).

---

## 📉 The "Traffic Cop" (Sequential AI Kernel)

To fit 3 Giants (LLM, VLM, ASR) into a small room (RAM), I built strict traffic rules in `ModelLifecycleManager.kt`:

1.  **State: LOAD_EYE** -> Clean RAM -> Load Vision -> Inference -> **UNLOAD**.
2.  **State: LOAD_EAR** -> Clean RAM -> Load ASR -> Transcribe -> **UNLOAD**.
3.  **State: LOAD_BRAIN** -> Clean RAM -> Load LLM -> Reason -> **UNLOAD**.

*Result: I never exceed 3.5GB Peak RAM usage, preventing the Android Low Memory Killer (LMK) from crashing the app during critical fieldwork.*

---

## 🛠️ Tech Stack & Requirements

*   **Language:** Kotlin (Android Native)
*   **AI Runtime:** Google LiteRT (TensorFlow Lite), ONNX Runtime
*   **Networking:** Android Nearby Connections API (P2P Mesh)
*   **Build System:** Gradle (Agp 8.2.0)
*   **Min SDK:** 26 (Android 8.0)
*   **Target Device:** 4GB RAM minimum (Pixel 4a or equivalent)

---

## 🔮 Future Roadmap

*   **Phase 2:** Integrate **MedGemma 4B** via knowledge distillation for high-end Anchor devices (8GB+ RAM).
*   **Phase 3:** Satellite uplink integration for Anchor nodes to sync with global health ministries.

---

## 🤝 Credits

*   **Google DeepMind & Kaggle:** For the MedGemma weights and the challenge.
*   **Sherpa ONNX:** For the model.
*   **Open Source Community:** For the constant drive to democratize AI.

---

## 📜 License & Open Source Commitment

**Current License: Apache License 2.0**  
This project is licensed under the Apache License 2.0, which permits open-source sharing, modification, and commercial use while providing strong patent protections.

**MedGemma Impact Challenge Commitment:** If MedGemma Forensic is selected as a winning submission, I formally agree to re-license the original application code under Creative Commons Attribution 4.0 International (CC BY 4.0) as required by competition rules.

**Note:** Integrated model weights including MedSigLIP, Gemma 3, and MedASR base models remain subject to their respective Google HAI-DEF and original upstream licenses and are exempt from this re-licensing commitment per competition rules Section 2.5.
