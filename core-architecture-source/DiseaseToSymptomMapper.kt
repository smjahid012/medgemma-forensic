package com.medgemma.forensic.data

import android.util.Log

/**
 * Disease to Symptom Mapper - Bridges MedSigLIP classifications to AnomalyEngine symptoms.
 * 
 * This mapper takes disease classifications from MedSigLIP (vision model) and maps them
 * to the structured symptom vocabulary expected by AnomalyEngine.
 * 
 * This is crucial for the anomaly detection pipeline:
 * 1. MedSigLIP classifies image → outputs disease label (e.g., "Dengue Fever")
 * 2. DiseaseToSymptomMapper maps disease → symptoms
 * 3. AnomalyEngine analyzes symptoms → detects outbreak patterns
 * 
 * Usage in pipeline:
 * - When MedSigLIP returns a classification, use this to convert to symptoms
 * - Combine with VoiceSymptomExtractor results for complete symptom picture
 * - Pass combined symptoms to AnomalyEngine for anomaly detection
 */
object DiseaseToSymptomMapper {

    private const val TAG = "DiseaseToSymptomMapper"

    /**
     * Disease to symptoms mapping based on WHO clinical case definitions
     * and common infectious disease presentations.
     */
    private val diseaseToSymptoms = mapOf(
        // Hemorrhagic Fevers
        "Dengue Fever" to listOf("Fever", "Headache", "Muscle_Pain", "Bleeding_Gums", "Rash_Petechiae"),
        "Dengue Hemorrhagic Fever" to listOf("Fever", "Bleeding_Gums", "Rash_Petechiae", "Vomiting", "Low_Blood_Pressure"),
        "Ebola" to listOf("Fever", "Bleeding_Gums", "Vomiting", "Diarrhea", "Fatigue"),
        "Marburg Virus" to listOf("Fever", "Bleeding_Gums", "Rash_Petechiae", "Vomiting", "Confusion"),
        "Lassa Fever" to listOf("Fever", "Bleeding_Gums", "Vomiting", "Diarrhea", "Facial_Swelling"),
        
        // Vector-Borne
        "Malaria" to listOf("Fever", "Headache", "Muscle_Pain", "Fatigue", "Chills"),
        "Cholera" to listOf("Diarrhea", "Vomiting", "Dehydration", "Low_Blood_Pressure", "Muscle_Cramps"),
        "Typhoid Fever" to listOf("Fever", "Headache", "Abdominal_Pain", "Diarrhea", "Fatigue"),
        
        // Respiratory
        "COVID-19" to listOf("Fever", "Cough", "Breathing_Difficulty", "Fatigue", "Loss_of_Taste"),
        "Influenza" to listOf("Fever", "Cough", "Muscle_Pain", "Headache", "Fatigue"),
        "Tuberculosis" to listOf("Cough", "Fever", "Night_Sweats", "Weight_Loss", "Fatigue"),
        "Pneumonia" to listOf("Fever", "Cough", "Breathing_Difficulty", "Chest_Pain", "Fatigue"),
        "SARS" to listOf("Fever", "Cough", "Breathing_Difficulty", "Muscle_Pain", "Diarrhea"),
        "MERS" to listOf("Fever", "Cough", "Breathing_Difficulty", "Muscle_Pain", "Diarrhea"),
        
        // Childhood Diseases
        "Measles" to listOf("Fever", "Rash_Petechiae", "Cough", "Conjunctivitis", "Lymph_Nodes_Swelling"),
        "Chickenpox" to listOf("Rash_Petechiae", "Skin_Lesions", "Fever", "Fatigue", "Muscle_Pain"),
        "Mumps" to listOf("Fever", "Lymph_Nodes_Swelling", "Fatigue", "Muscle_Pain", "Headache"),
        
        // Skin Conditions
        "Monkeypox" to listOf("Fever", "Rash_Petechiae", "Skin_Lesions", "Lymph_Nodes_Swelling", "Fatigue"),
        "Herpes" to listOf("Skin_Lesions", "Rash_Petechiae", "Pain", "Fatigue", "Fever"),
        "Leprosy" to listOf("Skin_Lesions", "Numbness", "Muscle_Weakness", "Skin_Necrosis", "Fatigue"),
        
        // Neurological
        "Meningitis" to listOf("Fever", "Headache", "Neck_Stiffness", "Confusion", "Seizures"),
        "Encephalitis" to listOf("Fever", "Headache", "Confusion", "Seizures", "Neck_Stiffness"),
        "Rabies" to listOf("Confusion", "Seizures", "Fear_of_Water", "Muscle_Pain", "Fatigue"),
        
        // Gastrointestinal
        "Rotavirus" to listOf("Diarrhea", "Vomiting", "Fever", "Dehydration", "Abdominal_Pain"),
        "Hepatitis A" to listOf("Jaundice", "Fever", "Fatigue", "Abdominal_Pain", "Nausea"),
        "Hepatitis B" to listOf("Jaundice", "Fatigue", "Muscle_Pain", "Fever", "Loss_of_Appetite"),
        "Hepatitis E" to listOf("Jaundice", "Fever", "Fatigue", "Nausea", "Abdominal_Pain"),
        
        // MedSigLIP Visual Findings Mapping
        // Necrosis / Eschar / Gangrene
        "necrotic skin lesion" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain"),
        "skin necrosis" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain"),
        "necrotic ulcer" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain", "Unusual_Lesion_Color"),
        "ulcer with necrotic tissue" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain"),
        "eschar" to listOf("Skin_Necrosis", "Skin_Lesions", "Unusual_Lesion_Color"),
        "black eschar" to listOf("Skin_Necrosis", "Skin_Lesions", "Unusual_Lesion_Color"),
        "gangrenous tissue" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain", "Unusual_Lesion_Color"),
        "gangrene" to listOf("Skin_Necrosis", "Skin_Lesions", "Unusual_Lesion_Color"),
        "necrotizing infection" to listOf("Skin_Necrosis", "Skin_Lesions", "Fever", "Pain"),
        "necrotic wound" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain"),
        "dead tissue in wound" to listOf("Skin_Necrosis", "Skin_Lesions"),
        "ischemic skin necrosis" to listOf("Skin_Necrosis", "Skin_Lesions", "Pain"),
        "dry gangrene" to listOf("Skin_Necrosis", "Unusual_Lesion_Color"),
        "wet gangrene" to listOf("Skin_Necrosis", "Unusual_Lesion_Color", "Fever"),
        
        // Ulcers
        "skin ulcer" to listOf("Skin_Lesions", "Pain"),
        "chronic skin ulcer" to listOf("Skin_Lesions", "Pain"),
        "infected ulcer" to listOf("Skin_Lesions", "Fever", "Pain", "Unusual_Lesion_Color"),
        "open skin ulcer" to listOf("Skin_Lesions", "Pain", "Bleeding_Gums"), // Potential bleeding
        "deep skin ulcer" to listOf("Skin_Lesions", "Pain"),
        "pressure ulcer" to listOf("Skin_Lesions", "Pain"),
        "diabetic foot ulcer" to listOf("Skin_Lesions", "Numbness"), // Often painless/numb
        "venous ulcer" to listOf("Skin_Lesions", "Pain", "Lymph_Nodes_Swelling"),
        "arterial ulcer" to listOf("Skin_Lesions", "Pain"),
        "non healing ulcer" to listOf("Skin_Lesions", "Pain"),
        "ulcerated lesion" to listOf("Skin_Lesions", "Pain"),
        "ulcer with slough" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        
        // Infections
        "bacterial skin infection" to listOf("Skin_Lesions", "Fever", "Pain", "Unusual_Lesion_Color"),
        "infected skin lesion" to listOf("Skin_Lesions", "Fever", "Pain", "Unusual_Lesion_Color"),
        "infected wound" to listOf("Skin_Lesions", "Fever", "Pain"),
        "purulent skin infection" to listOf("Skin_Lesions", "Fever", "Unusual_Lesion_Color"), // Pus implies color change
        "cellulitis" to listOf("Skin_Lesions", "Fever", "Pain", "Unusual_Lesion_Color"), // Redness
        "abscess" to listOf("Skin_Lesions", "Fever", "Pain", "Unusual_Lesion_Color"),
        "skin abscess" to listOf("Skin_Lesions", "Fever", "Pain"),
        "infected necrotic wound" to listOf("Skin_Necrosis", "Skin_Lesions", "Fever", "Pain"),
        "inflamed skin lesion" to listOf("Skin_Lesions", "Pain", "Unusual_Lesion_Color"), // Redness
        "severe skin infection" to listOf("Skin_Lesions", "Fever", "Pain", "Lymph_Nodes_Swelling"),
        
        // Scaling / Crusting
        "flaky skin lesion" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "scaly skin lesion" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "crusted skin lesion" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "hyperkeratotic lesion" to listOf("Skin_Lesions"),
        "desquamating skin" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "dry scaly plaque" to listOf("Skin_Lesions"),
        "crusted ulcer" to listOf("Skin_Lesions", "Pain"),
        "scaly wound edge" to listOf("Skin_Lesions"),
        
        // Fungal
        "fungal skin infection" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "cutaneous fungal infection" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "tinea infection" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "ringworm lesion" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        "fungal plaque" to listOf("Skin_Lesions"),
        "dermatophyte infection" to listOf("Skin_Lesions", "Unusual_Lesion_Color"),
        
        // RASHES / CHICKENPOX (NEW for reduced 25 labels)
        "blister" to listOf("Skin_Lesions", "Rash_Petechiae", "Fever"),  // Chickenpox-like
        "vesicular rash" to listOf("Skin_Lesions", "Rash_Petechiae", "Fever"),  // Chickenpox
        "red skin lesion" to listOf("Skin_Lesions", "Rash_Petechiae", "Eye_Redness"),  // Measles-like
        
        // NORMAL / BENIGN CONTROLS (NEW for reduced 25 labels)
        "normal skin" to listOf(),
        "healthy skin" to listOf(),
        "healed skin" to listOf(),
        "benign skin lesion" to listOf(),
        "scar tissue" to listOf(),
        
        // MALARIA-RELATED VISUAL (NEW for reduced 25 labels)
        "fever with rash" to listOf("Fever", "Rash_Petechiae"),
        "skin redness" to listOf("Rash_Petechiae", "Fever"),
        "pale skin" to listOf("Jaundice", "Fatigue"),
        "fatigued appearance" to listOf("Fatigue"),
        "sweating skin" to listOf("Fever"),
        
        // CHICKENPOX SYMPTOMS (these are already symptom names, map to themselves)
        "Rash_Petechiae" to listOf("Rash_Petechiae"),
        "Skin_Lesions" to listOf("Skin_Lesions"),
        "blister" to listOf("Skin_Lesions", "Rash_Petechiae", "Fever"),
        "vesicular rash" to listOf("Skin_Lesions", "Rash_Petechiae", "Fever"),

        // Other / Unknown Patterns
        "Unknown Anomaly" to listOf("Fever", "Rash_Petechiae", "Fatigue", "Muscle_Pain"),
        "Normal" to listOf(),  // No symptoms for normal cases
        "Unclassified" to listOf("Fatigue", "Fever"),  // Default symptoms
    )

    /**
     * Map a MedSigLIP disease classification to AnomalyEngine symptoms.
     * 
     * @param diseaseClassification The disease label from MedSigLIP (e.g., "Dengue Fever")
     * @param confidence Confidence score from MedSigLIP (0.0-1.0)
     * @return List of mapped symptoms from AnomalyEngine vocabulary
     */
    fun mapDiseaseToSymptoms(diseaseClassification: String, confidence: Float = 0.5f): List<String> {
        if (diseaseClassification.isBlank()) {
            Log.w(TAG, "Empty disease classification, returning empty symptoms")
            return emptyList()
        }

        // Exact match first
        val symptoms = diseaseToSymptoms[diseaseClassification]
        
        if (symptoms != null) {
            Log.d(TAG, "Mapped '$diseaseClassification' to symptoms: $symptoms (confidence: $confidence)")
            return symptoms
        }
        
        // Partial match - check if any key is contained in the classification
        val partialMatch = diseaseToSymptoms.entries.find { (disease, _) ->
            diseaseClassification.contains(disease, ignoreCase = true) ||
            disease.contains(diseaseClassification, ignoreCase = true)
        }
        
        if (partialMatch != null) {
            Log.d(TAG, "Partial matched '$diseaseClassification' → '${partialMatch.key}' = ${partialMatch.value}")
            return partialMatch.value
        }
        
        // Default fallback for unknown diseases - use confidence to determine symptom severity
        Log.w(TAG, "No mapping found for '$diseaseClassification', using default symptoms")
        return if (confidence > 0.7f) {
            // High confidence unknown disease - include more symptoms for thorough analysis
            listOf("Fever", "Fatigue", "Muscle_Pain", "Rash_Petechiae")
        } else {
            // Low confidence - use basic symptoms
            listOf("Fever", "Fatigue")
        }
    }

    /**
     * Map disease to symptoms with confidence weighting.
     * 
     * Higher confidence diseases will have more complete symptom lists.
     * 
     * @param diseaseClassification The disease label from MedSigLIP
     * @param confidence Confidence score from MedSigLIP
     * @return Map of symptom to confidence score
     */
    fun mapDiseaseToSymptomsWithConfidence(diseaseClassification: String, confidence: Float): Map<String, Float> {
        val symptoms = mapDiseaseToSymptoms(diseaseClassification, confidence)
        
        // Weight each symptom by disease confidence
        return symptoms.associateWith { confidence }
    }

    /**
     * Get all supported diseases.
     */
    fun getSupportedDiseases(): List<String> = diseaseToSymptoms.keys.toList()

    /**
     * Check if a disease classification is supported.
     */
    fun isSupported(diseaseClassification: String): Boolean {
        return diseaseToSymptoms.containsKey(diseaseClassification) ||
               diseaseToSymptoms.keys.any { it.contains(diseaseClassification, ignoreCase = true) }
    }

    /**
     * Example mappings:
     * 
     * Input: ("Dengue Fever", 0.85f)
     * Output: ["Fever", "Headache", "Muscle_Pain", "Bleeding_Gums", "Rash_Petechiae"]
     * 
     * Input: ("Unknown Anomaly", 0.6f)  
     * Output: ["Fever", "Rash_Petechiae", "Fatigue", "Muscle_Pain"]
     * 
     * Input: ("Normal", 0.95f)
     * Output: []
     */
}
