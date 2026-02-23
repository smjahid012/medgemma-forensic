package com.medgemma.forensic.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * The "Traffic Cop" for AI Models.
 * Ensures strict sequential loading of heavy models to prevent OOM.
 */
object ModelLifecycleManager {

    enum class ModelState {
        IDLE,
        LOADING_EYE, RUNNING_EYE, 
        LOADING_EYE_TEXT, RUNNING_EYE_TEXT,
        LOADING_EAR, RUNNING_EAR,
        LOADING_SMALL_BRAIN, RUNNING_SMALL_BRAIN,
        LOADING_BIG_BRAIN, RUNNING_BIG_BRAIN,
        ERROR
    }

    private val _currentState = MutableStateFlow(ModelState.IDLE)
    val currentState: StateFlow<ModelState> = _currentState.asStateFlow()

    private var currentLoadedModel: Any? = null

    // --- HELPER: UNLOAD EVERYTHING ---
    // --- HELPER: UNLOAD EVERYTHING ---
    fun unloadAllAndGc() {
        // Force Garbage Collection
        currentLoadedModel = null
        MedSigLIPManager.close()
        MedASRManager.close()
        MedGemmaManager.close()
        
        System.gc() 
        Thread.sleep(100) // Give GC a moment
    }

    // --- 1. THE EYE (MedSigLIP) ---
    suspend fun loadEye(context: Context) {
        if (_currentState.value == ModelState.RUNNING_EYE) return
        
        _currentState.value = ModelState.LOADING_EYE
        unloadAllAndGc() // Kill others
        
        // Architecture strict requirement: Check available memory
        val availMem = getAvailableMemoryMB(context)
        if (availMem < 200) {
           System.gc()
           Thread.sleep(200) // Give GC more time
           if (getAvailableMemoryMB(context) < 150) { // Hard floor lowered
               throw RuntimeException("Insufficient RAM for MedSigLIP! Available: ${getAvailableMemoryMB(context)}MB")
           }
        }
        
        try {
            MedSigLIPManager.init(context)
            _currentState.value = ModelState.RUNNING_EYE
        } catch (e: Exception) {
            e.printStackTrace()
            _currentState.value = ModelState.ERROR
        }
    }

    // --- 1b. THE EYE TEXT (MedSigLIP Text Encoder) ---
    // Note: This keeps the Vision encoder loaded for classification!
    suspend fun loadEyeText(context: Context) {
        if (_currentState.value == ModelState.RUNNING_EYE_TEXT) return
        
        _currentState.value = ModelState.LOADING_EYE_TEXT
        
        // Don't unload eye - we need both vision AND text encoder for classification!
        // Only unload other models (Ear, Brain)
        MedASRManager.close()
        MedGemmaManager.close()
        System.gc()
        Thread.sleep(50)
        
        // Architecture strict requirement: Check available memory
        val availMem = getAvailableMemoryMB(context)
        if (availMem < 150) {
           System.gc()
           Thread.sleep(200)
           if (getAvailableMemoryMB(context) < 100) {
               throw RuntimeException("Insufficient RAM for MedSigLIP Text! Available: ${getAvailableMemoryMB(context)}MB")
           }
        }
        
        try {
            // If eye is not loaded, load it first
            if (MedSigLIPManager.getVisionInterpreter() == null) {
                MedSigLIPManager.init(context)
            }
            // Now load text encoder (this keeps vision interpreter)
            MedSigLIPManager.initTextEncoder(context)
            _currentState.value = ModelState.RUNNING_EYE_TEXT
        } catch (e: Exception) {
            e.printStackTrace()
            _currentState.value = ModelState.ERROR
        }
    }

    fun unloadEyeText() {
        MedSigLIPManager.closeTextEncoder()
        if (_currentState.value == ModelState.RUNNING_EYE_TEXT) {
            _currentState.value = ModelState.IDLE
        }
    }

    // --- 2. THE EAR (MedASR) ---
    suspend fun loadEar(context: Context) {
        if (_currentState.value == ModelState.RUNNING_EAR) return

        _currentState.value = ModelState.LOADING_EAR
        unloadAllAndGc() 
        
        // Architecture strict requirement: Check available memory
        if (getAvailableMemoryMB(context) < 120) { // Ear is smaller (150MB), lower threshold
           System.gc()
           Thread.sleep(200) 
           if (getAvailableMemoryMB(context) < 100) {
               throw RuntimeException("Insufficient RAM for MedASR! Available: ${getAvailableMemoryMB(context)}MB")
           }
        }
        
        try {
            MedASRManager.init(context)
            _currentState.value = ModelState.RUNNING_EAR
        } catch (e: Exception) {
            e.printStackTrace()
            _currentState.value = ModelState.ERROR
        }
    }

    // --- 3. THE BIG BRAIN (MedGemma 4B) ---
    suspend fun loadBigBrain(context: Context) {
        if (_currentState.value == ModelState.RUNNING_BIG_BRAIN) return

        _currentState.value = ModelState.LOADING_BIG_BRAIN
        // FORCE AGGRESSIVE GC FOR BIG BRAIN
        unloadAllAndGc() 
        System.gc() 
        Thread.sleep(500) // Wait longer for big brain slot
        
        // Architecture strict requirement: Check available memory
        val available = getAvailableMemoryMB(context)
        // MedGemma 4B is ~3.6GB, but heavily compressed/quantized it might run on less, 
        // effectively 2GB RAM device implies ~1GB usable. 
        // The architecture doc says "1GB RAM constraint" for the WHOLE SYSTEM, 
        // but 4B model is huge. We trust the authorized quantization.
        // However, we must fail fast if < 500MB free even after GC.
        if (available < 500) {
           System.gc()
           if (getAvailableMemoryMB(context) < 400) {
               throw RuntimeException("CRITICAL: RAM too low for MedGemma 4B! Available: ${available}MB")
           }
        }
        
        try {
            MedGemmaManager.initBig(context)
            _currentState.value = ModelState.RUNNING_BIG_BRAIN
        } catch (e: Exception) {
            e.printStackTrace()
            _currentState.value = ModelState.ERROR
        }
    }
    // --- 4. THE SMALL BRAIN (Gemma 1B) ---
    suspend fun loadSmallBrain(context: Context) {
        if (_currentState.value == ModelState.RUNNING_SMALL_BRAIN) return

        _currentState.value = ModelState.LOADING_SMALL_BRAIN
        // Small Brain is light (1GB), but we still unload others to be safe on low RAM
        unloadAllAndGc() 
        
        // Memory check for Small Brain
        if (getAvailableMemoryMB(context) < 300) {
           System.gc()
           Thread.sleep(200)
           if (getAvailableMemoryMB(context) < 270) {
               throw RuntimeException("Insufficient RAM for Gemma 270M! Available: ${getAvailableMemoryMB(context)}MB")
           }
        }
        
        try {
            MedGemmaManager.initSmall(context)
            _currentState.value = ModelState.RUNNING_SMALL_BRAIN
        } catch (e: Exception) {
            e.printStackTrace()
            _currentState.value = ModelState.ERROR
        }
    }
    
    private fun getAvailableMemoryMB(context: Context): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memoryInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.availMem / (1024 * 1024)
    }
}
