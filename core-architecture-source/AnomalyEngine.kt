package com.medgemma.forensic.data

import android.util.Log
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import kotlin.math.*

// Date formatting
import java.time.format.DateTimeFormatter

/**
 * Anomaly Detection Engine - Core Math for Pattern Recognition.
 * Ported from src/utils/anomaly_engine.py
 * 
 * ENHANCED FEATURES (FR21-FR22):
 * - Baseline Learning: Learn "normal" patterns per village/location
 * - Deviation Detection: Compare against historical baseline
 * - Enhanced Symptom Vocabulary: 25 symptoms for better novel pathogen detection
 * - Multi-Case Temporal Analysis: Detect outbreak spikes over weekly trends
 * 
 * TODO (Phase 3 - FR36-FR40): Multi-language support
 */

data class CaseVector(
    val caseId: String,
    val symptomVector: FloatArray,
    val location: Pair<Double, Double>?, // Lat, Lon
    val timestamp: LocalDateTime,
    val riskScore: Float = 0.0f,
    var clusterId: String? = null
)

data class AnomalyResult(
    val isAnomaly: Boolean,
    val anomalyScore: Float, // 0-1 confidence
    val clusterId: String?,
    val contributingFactors: List<String>,
    val temporalScore: Float,
    val spatialScore: Float,
    val patternDescription: String,
    val recommendedActions: List<String>,
    // FR21-FR22: Enhanced fields
    val deviationScore: Float = 0f,           // How unusual this case is for the location
    val baselineComparison: String = "",      // Comparison to normal for location
    val weeklyTrend: String = "NORMAL",      // OUTBREAK_SPIKE, ELEVATED, NORMAL, DECLINING
    val detectedSymptoms: List<String> = emptyList(),  // All detected symptoms
    val commonSymptomsForLocation: List<String> = emptyList()  // Normal symptoms for this area
)

object AnomalyEngine {

    private val cases = mutableListOf<CaseVector>()
    private val clusters = mutableMapOf<String, MutableList<CaseVector>>()

    // FR21-FR22: Village Baselines - Learn "normal" patterns per location
    // Key: Location string (village/area name), Value: Baseline statistics
    private val villageBaselines = mutableMapOf<String, BaselineStats>()

    // FR21-FR22: Enhanced symptom vocabulary for better novel pathogen detection
    // Expanded from 15 to 25 symptoms for comprehensive coverage
    private val SYMPTOM_VOCAB = listOf(
        // Original WHO standard symptoms
        "Fever", "Cough", "Bleeding_Gums", "Jaundice",
        "Weight_Loss", "Breathing_Difficulty", "Neck_Stiffness",
        "Rash_Petechiae", "Diarrhea", "Vomiting", "Headache",
        "Muscle_Pain", "Fatigue", "Chest_Pain", "Skin_Lesions",
        // FR21-FR22: Additional symptoms for novel pathogen detection
        "Lymph_Nodes_Swelling",  // HIV-like
        "Eye_Redness",           // Conjunctivitis
        "Nose_Bleeding",         // Hemorrhagic
        "Skin_Necrosis",         // Gangrene
        "Unusual_Lesion_Color",  // Unknown patterns
        "Rapid_Heart_Rate",      // Tachycardia
        "Low_Blood_Pressure",    // Shock
        "Confusion",             // Neurological
        "Seizures"              // Neurological
    )

    // Priority symptoms for cluster naming - clinically significant symptoms first
    // These are picked over generic symptoms like "Fever" for cluster naming
    private val PRIORITY_SYMPTOMS = listOf(
        "Skin_Necrosis",
        "Bleeding_Gums",
        "Rash_Petechiae",
        "Neck_Stiffness",
        "Breathing_Difficulty",
        "Confusion",
        "Seizures",
        "Jaundice",
        "Skin_Lesions"
    )

    // FR21-FR22: Baseline statistics for a location
    data class BaselineStats(
        val location: String,
        val avgCasesPerMonth: Float,
        val commonSymptoms: List<String>,  // What's normally seen in this area
        val seasonalFactors: Map<String, Float>,  // Summer vs Winter patterns
        val sampleSize: Int = 0,
        val lastUpdated: LocalDateTime = LocalDateTime.now()
    )

    // FR21-FR22: Trend analysis result
    data class TrendResult(
        val location: String,
        val trend: String,        // OUTBREAK_SPIKE, ELEVATED, NORMAL, DECLINING
        val casesThisWeek: Int,
        val casesLastMonth: Int,
        val trendRatio: Float,
        val recommendation: String
    )

    /**
     * Add a historical case to establish baseline.
     * Use this for loading historical data to populate the baseline.
     * 
     * This method:
     * - Adds the case to the case list
     * - Updates baseline (so historical cases become the "benchmark")
     * 
     * Use this for Phase 1: Load historical data
     */
    fun addHistoricalCase(
        caseId: String,
        symptoms: List<String>,
        location: Pair<Double, Double>? = null,
        timestamp: LocalDateTime = LocalDateTime.now(),
        riskScore: Float = 0.0f
    ) {
        val vector = symptomsToVector(symptoms)
        val case = CaseVector(caseId, vector, location, timestamp, riskScore)
        
        cases.add(case)
        recalculateClusters()
        
        // Update baseline - historical cases establish what is "normal"
        if (location != null) {
            Log.d("AnomalyEngine", "Historical case $caseId: Establishing baseline for ${location.first},${location.second}")
            updateBaseline(location, symptoms, timestamp)
        }
    }

    /**
     * Add a new case and analyze for anomalies.
     * 
     * This method:
     * - Adds the case to the case list
     * - Calculates deviation score (how unusual this case is)
     * - ONLY updates baseline if:
     *   1. A baseline ALREADY EXISTS for this location (not first case!)
     *   2. Deviation is LOW (< 0.3)
     * 
     * Use this for Phase 2: Analyzing new incoming cases
     */
    fun addCase(
        caseId: String,
        symptoms: List<String>,
        location: Pair<Double, Double>? = null,
        timestamp: LocalDateTime = LocalDateTime.now(),
        riskScore: Float = 0.0f
    ) {
        val vector = symptomsToVector(symptoms)
        val case = CaseVector(caseId, vector, location, timestamp, riskScore)
        
        cases.add(case)
        recalculateClusters()
        
        // FR21-FR22: Update baseline ONLY for normal cases (low deviation)
        // CRITICAL: Only update if baseline ALREADY exists - don't create from first case!
        // Uses spatial fallback to find nearby baselines
        if (location != null) {
            val locationKey = getLocationClusterKey(location)
            
            // Check if baseline exists (exact match OR nearby within 10km)
            val hasExactBaseline = villageBaselines.containsKey(locationKey)
            val hasNearbyBaseline = findNearestBaseline(location) != null
            val hasExistingBaseline = hasExactBaseline || hasNearbyBaseline
            
            // Calculate deviation score
            val deviationScore = calculateDeviationScore(case)
            
            // Only update baseline if:
            // 1. Baseline exists (exact OR nearby) AND
            // 2. Deviation is LOW (< 0.3) - meaning symptoms are common/normal
            if (hasExistingBaseline && deviationScore < 0.3f) {
                Log.d("AnomalyEngine", "Case $caseId has normal symptoms (deviation=$deviationScore) - updating baseline")
                updateBaseline(location, symptoms, timestamp)
            } else if (!hasExistingBaseline) {
                Log.d("AnomalyEngine", "Case $caseId: No baseline yet - NOT updating (need historical data first)")
            } else {
                Log.d("AnomalyEngine", "Case $caseId has unusual symptoms (deviation=$deviationScore) - NOT updating baseline")
            }
        }
    }

    // FR21-FR22: Update baseline statistics for a location
    private fun updateBaseline(location: Pair<Double, Double>, symptoms: List<String>, timestamp: LocalDateTime) {
        // Use location CLUSTER (rounded to ~1km) instead of exact coordinates
        // This groups nearby cases together instead of creating unique baselines for each case
        val locationKey = getLocationClusterKey(location)
        val existing = villageBaselines[locationKey]
        
        // Determine season based on month
        val month = timestamp.monthValue
        val season = when (month) {
            in 3..5 -> "SPRING"
            in 6..8 -> "SUMMER"
            in 9..11 -> "AUTUMN"
            else -> "WINTER"
        }
        
        if (existing == null) {
            // First case for this location - establish initial baseline
            villageBaselines[locationKey] = BaselineStats(
                location = locationKey,
                avgCasesPerMonth = 1f,
                commonSymptoms = symptoms,
                seasonalFactors = mapOf(season to 1f),
                sampleSize = 1,
                lastUpdated = timestamp
            )
        } else {
            // Update existing baseline with new data
            val newSampleSize = existing.sampleSize + 1
            val newCommonSymptoms = if (symptoms.isNotEmpty()) {
                // Update common symptoms (weighted towards new symptoms)
                val symptomCounts = (existing.commonSymptoms + symptoms)
                    .groupingBy { it }.eachCount()
                    .toList().sortedByDescending { it.second }
                    .take(5)
                    .map { it.first }
                symptomCounts
            } else {
                existing.commonSymptoms
            }
            
            // Update seasonal factors
            val newSeasonalFactors = existing.seasonalFactors.toMutableMap()
            newSeasonalFactors[season] = (newSeasonalFactors[season] ?: 0f) + 1f
            
            // Calculate new average (simple running average)
            val newAvgCases = (existing.avgCasesPerMonth * existing.sampleSize + 1f) / newSampleSize
            
            villageBaselines[locationKey] = BaselineStats(
                location = locationKey,
                avgCasesPerMonth = newAvgCases,
                commonSymptoms = newCommonSymptoms,
                seasonalFactors = newSeasonalFactors,
                sampleSize = newSampleSize,
                lastUpdated = timestamp
            )
        }
    }

    // FR21-FR22: Calculate deviation score - how unusual is this case for this location?
    // Includes spatial fallback: if no baseline in exact cluster, look for nearby baselines
    fun calculateDeviationScore(currentCase: CaseVector): Float {
        val location = currentCase.location ?: return 0f
        
        // First try exact cluster match
        var locationKey = getLocationClusterKey(location)
        var baseline = villageBaselines[locationKey]
        
        // SPATIAL FALLBACK: If no baseline in exact cluster, find nearest baseline within 10km
        if (baseline == null) {
            Log.d("AnomalyEngine", "No baseline for cluster $locationKey - searching nearby clusters...")
            val nearest = findNearestBaseline(location)
            if (nearest != null) {
                baseline = nearest.second
                locationKey = nearest.first
                Log.d("AnomalyEngine", "Found nearest baseline at $locationKey (${haversineDistance(location, Pair(locationKey.split(",")[0].toDouble(), locationKey.split(",")[1].toDouble()))}km away)")
            } else {
                return 0f // No baseline anywhere nearby
            }
        }
        
        // Get detected symptoms
        val detectedSymptoms = vectorToSymptoms(currentCase.symptomVector)
        
        if (detectedSymptoms.isEmpty()) return 0f
        
        // How many symptoms are NOT in the common symptoms for this location?
        val unusualSymptoms = detectedSymptoms.filter { it !in baseline.commonSymptoms }
        
        // Calculate unusualness ratio
        val unusualness = unusualSymptoms.size.toFloat() / detectedSymptoms.size.toFloat()
        
        // Boost score if we have many unusual symptoms (novel pathogen indicator)
        return min(1.0f, unusualness * 1.5f)
    }

    // FR21-FR22: Get baseline comparison string
    // Includes spatial fallback for when exact cluster has no baseline
    fun getBaselineComparison(case: CaseVector): String {
        val location = case.location ?: return "No baseline data for this location"
        
        // First try exact cluster match
        var locationKey = getLocationClusterKey(location)
        var baseline = villageBaselines[locationKey]
        
        // Spatial fallback: find nearest baseline within 10km
        if (baseline == null) {
            val nearest = findNearestBaseline(location)
            if (nearest != null) {
                baseline = nearest.second
                locationKey = nearest.first
            } else {
                return "No historical data for this area"
            }
        }
        
        val detectedSymptoms = vectorToSymptoms(case.symptomVector)
        val unusualSymptoms = detectedSymptoms.filter { it !in baseline.commonSymptoms }
        
        return buildString {
            append("Baseline: ~${String.format("%.1f", baseline.avgCasesPerMonth)} cases/month in this area. ")
            append("Common symptoms: ${baseline.commonSymptoms.joinToString(", ")}. ")
            if (unusualSymptoms.isNotEmpty()) {
                append("⚠️ UNUSUAL: ${unusualSymptoms.joinToString(", ")}")
            } else {
                append("✓ Symptoms match typical pattern for this location.")
            }
        }
    }

    // FR21-FR22: Multi-case temporal analysis - detect outbreak spikes
    fun analyzeWeeklyTrend(location: Pair<Double, Double>?, days: Int = 7): TrendResult {
        if (location == null) {
            return TrendResult(
                location = "Unknown",
                trend = "NORMAL",
                casesThisWeek = 0,
                casesLastMonth = 0,
                trendRatio = 0f,
                recommendation = "No location data available"
            )
        }
        
        val locationKey = getLocationClusterKey(location)
        val now = LocalDateTime.now()
        val weekAgo = now.minusDays(days.toLong())
        val monthAgo = now.minusDays(30L)
        
        // Cases in the last 7 days (using cluster key for matching)
        val clusterKey = getLocationClusterKey(location)
        val casesThisWeek = cases.count { 
            getLocationClusterKey(it.location ?: Pair(0.0, 0.0)) == clusterKey && it.timestamp.isAfter(weekAgo)
        }
        
        // Cases in the last 30 days
        val casesLastMonth = cases.count { 
            getLocationClusterKey(it.location ?: Pair(0.0, 0.0)) == clusterKey && it.timestamp.isAfter(monthAgo)
        }
        
        // Calculate trend ratio (this week vs average of last month)
        val weeklyBaseline = if (casesLastMonth > 0) casesLastMonth / 4f else 1f
        val trendRatio = if (weeklyBaseline > 0) casesThisWeek / weeklyBaseline else 0f
        
        val (trend, recommendation) = when {
            trendRatio >= 3.0 -> "OUTBREAK_SPIKE" to "🚨 CRITICAL: Immediate reporting to health authorities required!"
            trendRatio >= 2.0 -> "ELEVATED" to "⚠️ HIGH ALERT: Cases 2x normal - intensify surveillance"
            trendRatio >= 1.0 -> "NORMAL" to "✓ Normal activity for this period"
            trendRatio >= 0.5 -> "DECLINING" to "↓ Below average - continue monitoring"
            else -> "LOW" to "↓ Very low activity - possible data gap"
        }
        
        return TrendResult(
            location = locationKey,
            trend = trend,
            casesThisWeek = casesThisWeek,
            casesLastMonth = casesLastMonth,
            trendRatio = trendRatio,
            recommendation = recommendation
        )
    }

    // FR21-FR22: Get all baselines (for debugging/UI)
    fun getAllBaselines(): Map<String, BaselineStats> = villageBaselines.toMap()

    // FR21-FR22: Convert exact location to cluster key (~1km radius)
    // This groups nearby cases together instead of creating unique baselines for each case
    private fun getLocationClusterKey(location: Pair<Double, Double>): String {
        // Round to ~1km (0.01 degrees ≈ 1km)
        val latCluster = (location.first * 100).toInt() / 100.0
        val lonCluster = (location.second * 100).toInt() / 100.0
        return "${latCluster},${lonCluster}"
    }
    
    // FR21-FR22: Find nearest baseline within 10km radius
    // Used when exact cluster has no baseline (spatial fallback)
    private fun findNearestBaseline(location: Pair<Double, Double>): Pair<String, BaselineStats>? {
        var nearest: Pair<String, BaselineStats>? = null
        var minDist = Double.MAX_VALUE
        
        for ((key, baseline) in villageBaselines) {
            val parts = key.split(",")
            if (parts.size == 2) {
                val baselineLoc = Pair(parts[0].toDouble(), parts[1].toDouble())
                val dist = haversineDistance(location, baselineLoc)
                if (dist < minDist && dist <= 10.0) { // Within 10km
                    minDist = dist
                    nearest = Pair(key, baseline)
                }
            }
        }
        return nearest
    }

    private fun symptomsToVector(symptoms: List<String>): FloatArray {
        val vector = FloatArray(SYMPTOM_VOCAB.size)
        symptoms.forEach { symptom ->
            val idx = SYMPTOM_VOCAB.indexOf(symptom)
            if (idx != -1) {
                vector[idx] = 1.0f
            }
        }
        return vector
    }

    private fun recalculateClusters() {
        if (cases.size < 2) return

        clusters.clear()
        cases.forEach { case ->
            val clusterId = findNearestCluster(case)
            case.clusterId = clusterId
            
            clusters.getOrPut(clusterId) { mutableListOf() }.add(case)
        }
    }

    private fun findNearestCluster(case: CaseVector): String {
        if (clusters.isEmpty()) return "cluster_${clusters.size}"

        var bestCluster: String? = null
        var bestDistance = Float.MAX_VALUE

        clusters.forEach { (id, clusterCases) ->
            if (clusterCases.isNotEmpty()) {
                // Calculate average vector (centroid)
                val centroid = FloatArray(SYMPTOM_VOCAB.size)
                clusterCases.forEach { c ->
                    for (i in centroid.indices) {
                        centroid[i] += c.symptomVector[i]
                    }
                }
                for (i in centroid.indices) {
                    centroid[i] /= clusterCases.size.toFloat()
                }

                val dist = euclideanDistance(case.symptomVector, centroid)
                if (dist < bestDistance) {
                    bestDistance = dist
                    bestCluster = id
                }
            }
        }

        return bestCluster ?: "cluster_${clusters.size}"
    }

    private fun euclideanDistance(v1: FloatArray, v2: FloatArray): Float {
        var sum = 0.0
        for (i in v1.indices) {
            val diff = v1[i] - v2[i]
            sum += diff * diff
        }
        return sqrt(sum).toFloat()
    }

    fun analyzeAnomaly(
        caseId: String,
        lookbackDays: Int = 7,
        spatialRadiusKm: Double = 10.0
    ): AnomalyResult {
        val targetCase = cases.find { it.caseId == caseId } 
            ?: throw IllegalArgumentException("Case $caseId not found")

        val similarCases = getSimilarCases(targetCase, lookbackDays)

        val temporalScore = calculateTemporalScore(targetCase, similarCases)
        val spatialScore = calculateSpatialScore(targetCase, similarCases)
        val patternScore = calculatePatternScore(targetCase, similarCases)
        
        // FR21-FR22: Calculate deviation score (unusualness for location)
        val deviationScore = calculateDeviationScore(targetCase)
        
        // FR21-FR22: Get baseline comparison
        val baselineComparison = getBaselineComparison(targetCase)
        
        // FR21-FR22: Get weekly trend
        val trendResult = analyzeWeeklyTrend(targetCase.location)
        
        // Get all detected symptoms
        val detectedSymptoms = vectorToSymptoms(targetCase.symptomVector)
        
        // Get common symptoms for location
        val location = targetCase.location
        val commonForLocation = if (location != null) {
            val locationKey = getLocationClusterKey(location)
            villageBaselines[locationKey]?.commonSymptoms ?: emptyList()
        } else {
            emptyList()
        }
        
        // Custom Weighting with deviation score boost
        var anomalyScore = (temporalScore * 0.25f) + (spatialScore * 0.25f) + (patternScore * 0.3f) + (deviationScore * 0.2f)
        
        // Boost for outbreak spikes
        if (trendResult.trend == "OUTBREAK_SPIKE") {
            anomalyScore = min(1.0f, anomalyScore * 1.3f)
        } else if (trendResult.trend == "ELEVATED") {
            anomalyScore = min(1.0f, anomalyScore * 1.15f)
        }
        
        // Legacy boost for high temporal + spatial
        if (temporalScore > 0.7f && spatialScore > 0.7f) {
            anomalyScore = min(1.0f, anomalyScore * 1.2f)
        }

        // FR21-FR22: OUTBREAK_SPIKE should always be flagged as anomaly
        // Also flag high deviation cases (unusual symptoms) as anomalies
        val isAnomaly = when {
            trendResult.trend == "OUTBREAK_SPIKE" -> true  // Always flag outbreak spikes
            deviationScore > 0.3f -> true                  // Flag unusual symptom patterns
            anomalyScore > 0.6f -> true                    // Standard threshold
            else -> false
        }

        return AnomalyResult(
            isAnomaly = isAnomaly,
            anomalyScore = anomalyScore,
            clusterId = targetCase.clusterId,
            contributingFactors = getContributingFactors(targetCase, similarCases),
            temporalScore = temporalScore,
            spatialScore = spatialScore,
            patternDescription = generatePatternDescription(targetCase, similarCases, anomalyScore, trendResult),
            recommendedActions = generateRecommendations(anomalyScore, trendResult),
            // FR21-FR22: New enhanced fields
            deviationScore = deviationScore,
            baselineComparison = baselineComparison,
            weeklyTrend = trendResult.trend,
            detectedSymptoms = detectedSymptoms,
            commonSymptomsForLocation = commonForLocation
        )
    }

    private fun getSimilarCases(target: CaseVector, days: Int): List<CaseVector> {
        val cutoff = LocalDateTime.now().minusDays(days.toLong())
        return cases.filter { it.caseId != target.caseId && it.timestamp.isAfter(cutoff) && cosineSimilarity(target.symptomVector, it.symptomVector) > 0.5 }
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0.0f
        var norm1 = 0.0f
        var norm2 = 0.0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }
        if (norm1 == 0f || norm2 == 0f) return 0f
        return dot / (sqrt(norm1) * sqrt(norm2))
    }
    
    // --- SCORING FUNCTIONS ---

    private fun calculateTemporalScore(case: CaseVector, similar: List<CaseVector>): Float {
        if (similar.isEmpty()) return 0.0f
        
        val timeDiffs = similar.map { 
            abs(ChronoUnit.HOURS.between(case.timestamp, it.timestamp)).toFloat()
        }
        val avgDiff = timeDiffs.average().toFloat()
        
        // Within 24 hours = High Score
        return if (avgDiff < 24) min(1.0f, (24 - avgDiff) / 24 + 0.3f) else 0.2f
    }

    private fun calculateSpatialScore(case: CaseVector, similar: List<CaseVector>): Float {
        if (similar.isEmpty() || case.location == null) return 0.0f
        
        val distances = similar.mapNotNull { 
            it.location?.let { loc -> haversineDistance(case.location, loc) } 
        }
        
        if (distances.isEmpty()) return 0.0f
        val avgDist = distances.average().toFloat()
        
        // Within 10km = High Score
        return if (avgDist < 10) min(1.0f, (10 - avgDist) / 10 + 0.3f) else 0.1f
    }
    
    private fun calculatePatternScore(case: CaseVector, similar: List<CaseVector>): Float {
         if (similar.isEmpty()) return 0.0f
         // How much do symptoms overlap?
         val overlaps = similar.map { 
             // Simple dot product of binary vectors is intersection count
             var intersection = 0.0f
             var union = 0.0f // size of similar case symptoms
             for(i in case.symptomVector.indices) {
                 if(case.symptomVector[i] > 0 && it.symptomVector[i] > 0) intersection++
                 if(it.symptomVector[i] > 0) union++
             }
             if(union == 0f) 0f else intersection / union
         }
         return overlaps.average().toFloat()
    }

    private fun haversineDistance(loc1: Pair<Double, Double>, loc2: Pair<Double, Double>): Double {
        val R = 6371.0 // Radius of the earth in km
        val lat1 = Math.toRadians(loc1.first)
        val lon1 = Math.toRadians(loc1.second)
        val lat2 = Math.toRadians(loc2.first)
        val lon2 = Math.toRadians(loc2.second)

        val dLat = lat2 - lat1
        val dLon = lon2 - lon1

        val a = sin(dLat / 2).pow(2) + cos(lat1) * cos(lat2) * sin(dLon / 2).pow(2)
        val c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c
    }
    
    // --- GENERATORS ---

    private fun getContributingFactors(case: CaseVector, similar: List<CaseVector>): List<String> {
        val factors = mutableListOf<String>()
        if (similar.size >= 3) factors.add("High case count (${similar.size + 1} in cluster)")
        if (case.riskScore > 0.7) factors.add("Individual risk factors elevated")
        val symptoms = vectorToSymptoms(case.symptomVector)
        if ("Bleeding_Gums" in symptoms || "Rash_Petechiae" in symptoms) factors.add("Hemorrhagic symptoms present")
        return factors
    }

    private fun generatePatternDescription(case: CaseVector, similar: List<CaseVector>, score: Float, trendResult: TrendResult? = null): String {
        val symptoms = vectorToSymptoms(case.symptomVector)
        
        // Pick the most clinically significant symptom for cluster naming
        // Priority symptoms take precedence over generic ones like "Fever"
        val symptomStr = symptoms.firstOrNull { it in PRIORITY_SYMPTOMS }
                        ?: symptoms.firstOrNull()
                        ?: "Unknown"
        
        if (similar.isEmpty()) return "Isolated case with symptoms: $symptomStr"
        
        val severity = when {
            score > 0.8 -> "CRITICAL CLUSTER"
            score > 0.6 -> "POTENTIAL OUTBREAK"
            score > 0.4 -> "ELEVATED RISK"
            else -> "Notice"
        }
        
        // FR21-FR22: Add trend information
        val trendSuffix = if (trendResult != null && trendResult.trend != "NORMAL") {
            " [${trendResult.trend}: ${trendResult.casesThisWeek} cases this week vs ${String.format("%.1f", trendResult.casesLastMonth / 4f)} expected]"
        } else ""
        
        return "$severity: $symptomStr cluster detected with ${similar.size + 1} cases.$trendSuffix"
    }

    // FR21-FR22: Enhanced recommendations with trend awareness
    private fun generateRecommendations(score: Float, trendResult: TrendResult? = null): List<String> {
        // Trend-based recommendations take priority
        if (trendResult != null) {
            when (trendResult.trend) {
                "OUTBREAK_SPIKE" -> return listOf(
                    "🚨 CRITICAL: Report to health authorities IMMEDIATELY",
                    "Activate outbreak response protocol",
                    "Collect samples for laboratory confirmation",
                    "Implement patient isolation",
                    "Begin contact tracing"
                )
                "ELEVATED" -> return listOf(
                    "⚠️ HIGH ALERT: Intensify surveillance",
                    "Flag for review by epidemiologist",
                    "Follow-up within 24 hours",
                    "Check for additional cases in area"
                )
                "DECLINING" -> return listOf(
                    "Continue standard monitoring",
                    "Verify data completeness",
                    "Prepare weekly summary report"
                )
            }
        }
        
        // Score-based recommendations
        return when {
            score > 0.8 -> listOf("URGENT: Report to health authorities", "Samples required", "Isolate patient")
            score > 0.6 -> listOf("Flag for review", "Follow-up 24h", "Check surveillance DB")
            score > 0.4 -> listOf("Monitor symptoms", "Follow-up 48h")
            else -> listOf("Standard management")
        }
    }

    private fun vectorToSymptoms(vector: FloatArray): List<String> {
        val list = mutableListOf<String>()
        vector.forEachIndexed { index, fl -> 
            if (fl > 0) list.add(SYMPTOM_VOCAB[index])
        }
        return list
    }
    
    // --- HELPER FOR UI ---
    fun getSupportedSymptoms(): List<String> = SYMPTOM_VOCAB
    
    // FR21-FR22: Get historical case count for a location (using cluster key)
    fun getCaseCountForLocation(location: Pair<Double, Double>?): Int {
        if (location == null) return 0
        val clusterKey = getLocationClusterKey(location)
        return cases.count { getLocationClusterKey(it.location ?: Pair(0.0, 0.0)) == clusterKey }
    }
    
    // FR21-FR22: Get all cases sorted by timestamp
    fun getAllCases(): List<CaseVector> = cases.sortedByDescending { it.timestamp }
    
    // FR21-FR22: Get cluster information for pattern mapping
    fun getClusterInfo(): Map<String, Int> {
        return cases.groupBy { it.clusterId ?: "unknown" }
            .mapValues { it.value.size }
    }
    
    // FR21-FR22: Reset baseline - use this to clear polluted baselines
    // Call this when you want to start fresh (e.g., after app update or data reset)
    fun resetBaseline() {
        Log.d("AnomalyEngine", "🔄 RESETTING ALL BASELINES")
        villageBaselines.clear()
        cases.clear()
    }
    
    // FR21-FR22: Get baseline status for debugging
    fun hasBaselineForLocation(location: Pair<Double, Double>?): Boolean {
        if (location == null) return false
        val locationKey = getLocationClusterKey(location)
        return villageBaselines.containsKey(locationKey)
    }
}
