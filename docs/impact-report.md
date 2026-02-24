### Project name 
MedGemma FORENSIC

### Your team 
** Sm Jahid Bin Esha ** – Solo Developer. Responsible for the end-to-end architecture, Android application development (Kotlin/Jetpack Compose), Edge AI model integration (LiteRT/ONNX), P2P Mesh networking, and data science simulation.

### Problem statement
**The "Last Mile" Blind Spot in Pandemic Surveillance**

Traditional disease surveillance systems face a fundamental, fatal flaw: they rely on digitized data sources—news, social media, hospital records, and internet connectivity. This approach misses the 4.2 billion people living in the "Last Mile." In rural, underserved regions, by the time an outbreak reaches a digitized hospital system, it has already spread beyond containment. 

MedGemma FORENSIC addresses this urgent health equity gap. I provide an offline-first syndromic surveillance tool designed specifically for Village Health Workers (VHWs). Instead of attempting complex diagnoses at the edge, the system asks a more critical question: *"Is something statistically unusual happening here?"* By detecting deviations from localized, learned baselines, MedGemma FORENSIC identifies novel pathogens and localized outbreaks at "Patient Zero," potentially providing health agencies with a critical 7-14 day lead time in connectivity-dead zones.

### Overall solution: 
**The Autonomous, Offline Pathologist**

MedGemma FORENSIC acts as a field pathologist that requires zero internet connection. It is built strictly on HAI-DEF (Healthcare AI Framework) principles, utilizing a chain-of-evidence pipeline to synthesize multi-modal data into actionable intelligence.

1.  **Visual Evidence (The Eyes):** I leverage **Google's MedSigLIP 448** (TFLite) to analyze photos of skin lesions or physical conditions, matching them against a 25-dimension WHO-standard symptom vocabulary (e.g., identifying *Skin_Necrosis* vs. a standard rash).

2.  **Verbal Evidence (The Ears):** I integrate **Sherpa ONNX MedASR ** to securely transcribe verbal testimonies and clinical observations locally on the device, extracting critical keywords.

3.  **Reasoning & Intelligence (The Brain):** The visual findings, transcribed audio, and GPS coordinates are compiled into a Protobuf payload. **Gemma 3 1B (LiteRT)**, acting as the logic engine, processes this forensic dossier.

4.  **Pattern Hunter (The Engine):** An unsupervised **Spatiotemporal Anomaly Engine** compares the new case against a locally learned baseline. If symptoms deviate significantly (e.g., an unexpected cluster of hemorrhagic fever indicators), the system triggers an `OUTBREAK_SPIKE` or `ELEVATED RISK` alert.

### Technical details 
**Edge Feasibility & Resilient Architecture**

Deploying multiple heavy AI models on resource-constrained Android devices presents immense technical challenges. MedGemma FORENSIC solves this through a robust, hardware-aware architecture:

*   **Auto Role Assignment & Mesh Networking:** The application automatically detects device RAM. Devices with ≥ 7.1GB act as **Anchor Nodes** (running the full Gemma 3 1B pipeline). Standard phones (< 7.1GB RAM) act as **Worker Nodes** (sensors). I can reduce the ram size and fit it into 4 gigabyte ram. Workers capture visual/audio data and transmit serialized Protobuf payloads to the Anchor entirely offline via **Google Nearby Connections** (Bluetooth/WiFi Direct P2P mesh).

*   **Sequential "Traffic Cop" Memory Management:** To prevent Android's Out-Of-Memory (OOM) killer from terminating the app, I implemented a strict State Machine. The pipeline loads the model weights (totaling ~2.3 GB) *serially*: `MedSigLIP` → *Unload* → `MedASR` → *Unload* → `Gemma 3 1B`. At peak process usage, the system RAM stays well below Android limits, making it feasible for standard field devices.

*   **Clinical Trust Hierarchy & Priority Naming:** The system allows human-in-the-loop override. If the MedSigLIP vision model makes a low-confidence prediction, the user's verbal/manual confirmation overrides it, adding a severity boost to the Anomaly Score. Priority is given to hemorrhagic indicators (e.g., Bleeding_Gums, Skin_Necrosis) over generic symptoms (e.g., Fever), ensuring critical signals aren't diluted.

*   **Data Minimization (Privacy by Design):** The Anchor node performs the heavy lifting and stores the forensic dossier. The Worker node receives only a concise telemetry ping (e.g., `OUTBREAK_CONFIRMED`). Complete patient data and detailed reports never reside on the vulnerable Worker devices, drastically reducing the attack surface and aligning with GDPR/HIPAA principles.

### 🛡️ HAI-DEF Alignment & Open-Source Edge Engineering
MedGemma FORENSIC was built strictly following the **Healthcare AI Framework (HAI-DEF)**, prioritizing **Open-Weight Traceability** and robust edge deployment. We do not use closed, black-box cloud APIs. Every AI component executing on the device is fully transparent, locally executed, and required massive custom engineering to operate on a 4GB Android device:

1.  **Logic Engine:** **Google Gemma 3 1B** (LiteRT Edge-Optimized). Evaluated for transparent LLM reasoning. Traceable to Google's official model repositories and subject to the open Gemma License.

2.  **Vision Encoder:** **MedSigLIP 448**. This was not natively compatible with Android. I performed a custom TFLite conversion of Google Research's SigLIP architecture. To achieve clinical accuracy on Edge, I had to structurally rewrite the `SentencePieceTokenizer` in native Kotlin (achieving a 100% match against the 32,000-token HuggingFace ground truth without the heavy Python transformers library). I also engineered a custom L2 normalization and a `LOGIT_SCALE = 100.0f` scalar fix at the android  pipeline level to prevent softmax probability collapse when detecting subtle pathological signs like *'necrotic eschar'*.

3.  **Audio Transcription:** **Sherpa ONNX MedASR**. The official pre-compiled Android AAR binaries faced severe ABI mismatches with modern Kotlin compilers. To achieve stable clinical dictation, I bypassed the AAR wrappers entirely, extracting the raw C++ JNI `.so` libraries and compiling the engine natively from source code into the project's ecosystem.

By strictly utilizing open weights and engineering them for native edge execution, MedGemma FORENSIC ensures that clinical reviewers and public health authorities can definitively audit the inference logic, embedding spaces, and data processing pipeline without relying on proprietary vendor secrets.
