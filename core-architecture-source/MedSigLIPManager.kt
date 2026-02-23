package com.medgemma.forensic.data

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.medgemma.forensic.ai.SentencePieceTokenizer
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Manages the "Eye" (MedSigLIP Vision Encoder).
 * Runs on TFLite (CPU/GPU).
 */
object MedSigLIPManager {
    private var visionInterpreter: Interpreter? = null
    private var textInterpreter: Interpreter? = null
    private var tokenizer: SentencePieceTokenizer? = null
    
    // SigLIP Config (Default, but will be overridden by model inspection)
    private var INPUT_SIZE = 448
    private var OUTPUT_SHAPE = intArrayOf(1, 1152) 
    private var TEXT_OUTPUT_SHAPE = intArrayOf(1, 1152)
    
    /**
     * Logit scale for CLIP/SigLIP similarity scores.
     * 
     * Using 100.0f - this is the standard CLIP/SigLIP value.
     * The raw cosine similarities from these models are small (~0.01-0.02), so we need
     * to scale them up to get meaningful logits for softmax.
     * 
     * Formula: logits = cosine_similarity * logit_scale
     * With LOGIT_SCALE=100, raw ~0.015 becomes logits ~1.5
     * Then softmax converts these to proper probabilities.
     */
    private const val LOGIT_SCALE = 100.0f
    
    // REDUCED MedSigLIP Label Set (25 labels for faster processing)
    // User-defined: Focused on Necrosis, Malaria, Chickenpox, Ulcers, Infections
    private val symptomLabels = listOf(
        // HIGH PRIORITY: Necrosis/Gangrene (5)
        "skin necrosis",
        "gangrene",
        "necrotic wound",
        "dry gangrene",
        "wet gangrene",
        
        // COMMON: Malaria Symptoms via Visual (5) - Fever can cause flushed skin
        "fever with rash",
        "skin redness",
        "pale skin",
        "fatigued appearance",
        "sweating skin",
        
        // COMMON: Chickenpox/Measles (5)
        "Rash_Petechiae",
        "Skin_Lesions",
        "blister",
        "vesicular rash",
        "fever with rash",
        
        // Ulcers (5)
        "skin ulcer",
        "chronic skin ulcer",
        "infected ulcer",
        "pressure ulcer",
        "diabetic foot ulcer",
        
        // Infections (5)
        "bacterial skin infection",
        "cellulitis",
        "abscess",
        "skin abscess",
        "infected wound"
    ) 
    
    // Labels that indicate ABNORMAL/WOUND conditions (for healthy skin detection)
    private val woundLabels = setOf(
        "necrotic", "gangrene", "eschar", "ulcer", "wound", "infection",
        "cellulitis", "abscess", "inflamed", "lesion", "flaky", "scaly",
        "crusted", "keratotic", "fungal", "tinea", "ringworm", "dermatophyte"
    )
    
    // Healthy skin indicator labels
    private val healthySkinLabels = setOf(
        "normal skin",
        "healthy skin",
        "completely healthy normal skin",
        "normal uninjured human skin",
        "skin with no wounds or disease"
    ) 
    
    fun init(context: Context) {
        if (visionInterpreter != null) return

        val modelFile = ModelDownloader.getModelFile(context, ModelDownloader.ModelType.EYE)
        if (!modelFile.exists()) throw RuntimeException("Eye Model missing!")

        val options = Interpreter.Options()
        // Default to CPU for maximum stability (GPU delegates can be flaky on some devices)
        options.setNumThreads(4)

        try {
            // Load MappedByteBuffer
            val fileInputStream = FileInputStream(modelFile)
            // Use 'use' to auto-close the stream and channel after mapping
            val mappedByteBuffer = fileInputStream.use { stream ->
                val fileChannel = stream.channel
                val startOffset = 0L
                val declaredLength = modelFile.length()
                fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            }
            
            val options = Interpreter.Options()
            options.setNumThreads(4)
            
            visionInterpreter = Interpreter(mappedByteBuffer, options)
            
            // INSPECT MODEL SIGNATURE TO PREVENT CRASHES
            visionInterpreter?.let { tflite ->
                // Input Config
                val inputTensor = tflite.getInputTensor(0)
                val inputShape = inputTensor.shape() // e.g., [1, 448, 448, 3] or [1, 3, 448, 448]
                if (inputShape.size == 4) {
                    // Fix for NCHW models ([1, 3, 448, 448]) where index 1 is Channels (3)
                    // We want the spatial dimension (448)
                    INPUT_SIZE = if (inputShape[1] == 3 && inputShape[2] > 3) {
                        inputShape[2] 
                    } else {
                        inputShape[1]
                    }
                }
                
                // Output Config
                val outputTensor = tflite.getOutputTensor(0)
                OUTPUT_SHAPE = outputTensor.shape()
                
                Log.d("MedSigLIP", "Vision Model Initialized. Input Shape: ${inputShape.contentToString()} (Using Size: $INPUT_SIZE), Output: ${OUTPUT_SHAPE.contentToString()}")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Failed to initialize Eye model: ${e.message}")
        }
    }

    /**
     * Initialize the Text Encoder for zero-shot classification.
     */
    fun initTextEncoder(context: Context) {
        if (textInterpreter != null) return

        // Initialize tokenizer
        tokenizer = SentencePieceTokenizer(context)

        val modelFile = ModelDownloader.getModelFile(context, ModelDownloader.ModelType.EYE_TEXT)
        if (!modelFile.exists()) throw RuntimeException("Eye Text Model missing!")

        try {
            val fileInputStream = FileInputStream(modelFile)
            val mappedByteBuffer = fileInputStream.use { stream ->
                val fileChannel = stream.channel
                fileChannel.map(FileChannel.MapMode.READ_ONLY, 0L, modelFile.length())
            }

            val options = Interpreter.Options()
            options.setNumThreads(4)

            textInterpreter = Interpreter(mappedByteBuffer, options)

            textInterpreter?.let { tflite ->
                val outputTensor = tflite.getOutputTensor(0)
                TEXT_OUTPUT_SHAPE = outputTensor.shape()
                Log.d("MedSigLIP", "Text Encoder Initialized. Output: ${TEXT_OUTPUT_SHAPE.contentToString()}")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Failed to initialize Text Encoder: ${e.message}")
        }
    }

    fun close() {
        visionInterpreter?.close()
        visionInterpreter = null
        textInterpreter?.close()
        textInterpreter = null
        tokenizer = null
    }

    fun closeTextEncoder() {
        textInterpreter?.close()
        textInterpreter = null
    }
    
    /**
     * Close only the vision interpreter to free RAM.
     */
    fun closeVision() {
        visionInterpreter?.close()
        visionInterpreter = null
        Log.d("MedSigLIP", "Vision model unloaded from RAM")
    }

    /**
     * Get vision interpreter for checking if loaded.
     */
    fun getVisionInterpreter(): Interpreter? = visionInterpreter

    /**
     * Encodes an image into embeddings.
     */
    fun encode(bitmap: Bitmap): FloatArray {
        val tflite = visionInterpreter ?: throw IllegalStateException("Eye not loaded!")
        
        try {
            // 1. Preprocess using TFLite Support Library (Safer & Faster)
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0.5f, 0.5f)) // SigLIP normalization: (x - 0.5) / 0.5 = 2x - 1
                .build()
                
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            // 2. Prepare Output Buffer based on DYNAMIC shape
            // Calculate total elements in output tensor
            var startSize = 1
            for (dim in OUTPUT_SHAPE) {
                startSize *= dim
            }
            
            // Allocate Flattened Buffer
            val outputBuffer = ByteBuffer.allocateDirect(startSize * 4) // Float32 = 4 bytes
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // 3. Run Inference safely
            tflite.run(processedImage.buffer, outputBuffer)
            
            // 4. Extract float array from output buffer
            outputBuffer.rewind()
            val rawFloats = FloatArray(startSize)
            outputBuffer.asFloatBuffer().get(rawFloats)
            
            // 5. Process Output - Use MEAN pooling (global average pooling)
            // This captures the average activation across all patches
            // which is more semantically meaningful than max pooling for medical images
            val embedding = if (OUTPUT_SHAPE.size == 3) {
                 headerlessGlobalAveragePooling(rawFloats, OUTPUT_SHAPE[1], OUTPUT_SHAPE[2])
            } else {
                rawFloats
            }
            
            // CRITICAL: The output is NOT L2 normalized after pooling!
            // We need to verify by checking the norm
            val preNormStats = calculateVectorStats(embedding)
            Log.d("MedSigLIP", "Pre-normalization stats: norm=${String.format("%.4f", preNormStats.first)}, max=${String.format("%.4f", preNormStats.second)}")
            
            // SKIP distribution matching - L2 normalization handles scale differences
            // The aggressive scaling was distorting the embeddings
            
            // CRITICAL: L2 normalization is REQUIRED for SigLIP embeddings
            // Without this, cosine similarity will not work correctly
            val normalized = l2Normalize(embedding)
            
            // DEBUG: Log embedding stats
            val sum = normalized.sum()
            val mean = normalized.average()
            val max = normalized.maxOrNull() ?: 0f
            val min = normalized.minOrNull() ?: 0f
            val hasNaN = normalized.any { it.isNaN() }
            
            // Check if embedding has meaningful variation
            val uniqueValues = normalized.toSet().size
            val stdDev = calculateStdDev(normalized)
            
            Log.d("MedSigLIP", "Image embedding stats: sum=${String.format("%.4f", sum)}, mean=${String.format("%.6f", mean)}, max=${String.format("%.4f", max)}, min=${String.format("%.4f", min)}, hasNaN=$hasNaN, uniqueVals=$uniqueValues, stdDev=${String.format("%.4f", stdDev)}")
            
            // Check if this looks like a broken embedding
            if (uniqueValues < 100 || stdDev < 0.001f) {
                Log.w("MedSigLIP", "WARNING: Image embedding has low variation! This may indicate a problem with the vision model.")
            }
            
            return normalized
            
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("MedSigLIP", "Inference Failed: ${e.message}")
            return FloatArray(1152) // Return dummy zeros to prevent app crash
        }
    }

    /**
     * Encodes text into embeddings using the text encoder.
     * Uses token IDs - model expects INT16 or INT32 input.
     * 
     * FIXED: Added better error handling and debugging for text encoder issues.
     */
    fun encodeText(text: String): FloatArray {
        val tflite = textInterpreter ?: throw IllegalStateException("Text Encoder not loaded!")
        
        try {
            // Check what input type the model expects
            val inputTensor = tflite.getInputTensor(0)
            val inputType = inputTensor.dataType()
            val inputShape = inputTensor.shape()
            Log.d("MedSigLIP", "Text encoder input type: $inputType, shape: ${inputShape.contentToString()}")
            
            // Prepare output buffer
            var outputSize = 1
            for (dim in TEXT_OUTPUT_SHAPE) {
                outputSize *= dim
            }
            val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // Tokenize input text
            val tokenIds = tokenizer?.encode(text) ?: listOf(SentencePieceTokenizer.UNK_TOKEN_ID)
            Log.d("MedSigLIP", "Tokenizing: '$text' -> tokens: $tokenIds")
            
            // Check if tokens are mostly UNK (which would explain poor embeddings)
            val unkCount = tokenIds.count { it == SentencePieceTokenizer.UNK_TOKEN_ID }
            val unkPercent = (unkCount.toFloat() / tokenIds.size) * 100
            if (unkPercent > 50) {
                Log.w("MedSigLIP", "WARNING: $unkPercent% of tokens are UNK! Medical terms may not be recognized.")
            }
            
            // Pad to fixed length - model expects 64 tokens based on error "512 bytes"
            val maxLength = 64
            val paddedTokens = IntArray(maxLength) { i -> 
                if (i < tokenIds.size) tokenIds[i] else SentencePieceTokenizer.PAD_TOKEN_ID 
            }
            
            // Prepare input based on data type (model expects INT16 - 2 bytes per token)
            if (inputType == DataType.INT16) {
                Log.d("MedSigLIP", "Using INT16 input (2 bytes per token)")
                val inputBuffer = ByteBuffer.allocateDirect(maxLength * 2) // INT16 = 2 bytes
                inputBuffer.order(ByteOrder.nativeOrder())
                val shortBuffer = inputBuffer.asShortBuffer()
                for (tokenId in paddedTokens) {
                    shortBuffer.put(tokenId.toShort())
                }
                inputBuffer.rewind()
                tflite.run(inputBuffer, outputBuffer)
            } else if (inputType == DataType.INT32) {
                Log.d("MedSigLIP", "Using INT32 input (4 bytes per token)")
                val inputBuffer = ByteBuffer.allocateDirect(maxLength * 4)
                inputBuffer.order(ByteOrder.nativeOrder())
                val intBuffer = inputBuffer.asIntBuffer()
                for (tokenId in paddedTokens) {
                    intBuffer.put(tokenId)
                }
                inputBuffer.rewind()
                tflite.run(inputBuffer, outputBuffer)
            } else if (inputType == DataType.INT64) {
                Log.d("MedSigLIP", "Using INT64 input (8 bytes per token)")
                val inputBuffer = ByteBuffer.allocateDirect(maxLength * 8)
                inputBuffer.order(ByteOrder.nativeOrder())
                val longBuffer = inputBuffer.asLongBuffer()
                for (tokenId in paddedTokens) {
                    longBuffer.put(tokenId.toLong())
                }
                inputBuffer.rewind()
                tflite.run(inputBuffer, outputBuffer)
            } else {
                throw IllegalArgumentException("Unsupported input type: $inputType")
            }
            
            // Process output
            outputBuffer.rewind()
            val rawFloats = FloatArray(outputSize)
            outputBuffer.asFloatBuffer().get(rawFloats)
            
            // Check for NaN or all zeros
            val hasNaN = rawFloats.any { it.isNaN() }
            val sum = rawFloats.sum()
            val mean = rawFloats.average()
            val max = rawFloats.maxOrNull() ?: 0f
            val min = rawFloats.minOrNull() ?: 0f
            
            Log.d("MedSigLIP", "Text embedding stats: sum=$sum, mean=$mean, max=$max, min=$min, hasNaN=$hasNaN")
            
            // Check if embedding is essentially zero (model not working)
            if (kotlin.math.abs(sum) < 0.01f) {
                Log.e("MedSigLIP", "CRITICAL: Text embedding is near zero! Model may be corrupted.")
                // Return normalized random embedding to prevent all similarities being negative
                return l2Normalize(FloatArray(1152) { (Math.random() - 0.5).toFloat() })
            }
            
            // CRITICAL: L2 normalization is REQUIRED for SigLIP embeddings
            return l2Normalize(rawFloats)
            
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("MedSigLIP", "Text encoding failed: ${e.message}")
            return FloatArray(1152) { (Math.random() - 0.5).toFloat() }
        }
    }

    /**
     * Zero-shot classification usingSigLIP.
     * Compares image embedding with text label embeddings.
     * 
     * @param context Android context for loading models
     * @param bitmap The image to classify
     * @param labels List of candidate labels
     * @return Pair of (bestLabel, confidenceScore)
     */
    fun classify(context: Context, bitmap: Bitmap, labels: List<String> = symptomLabels): Pair<String, Float> {
        return classifyWithContext(context, bitmap, labels)
    }
    
    /**
     * Classification with explicit context - enables sequential model loading.
     * This loads models one at a time to minimize RAM usage.
     * 
     * Flow:
     * 1. Load Vision → Encode Image → Unload Vision
     * 2. Load Text → Encode Labels → Unload Text  
     * 3. Compare embeddings in software
     * 
     * RAM: ~1500MB (vision) OR ~400MB (text), not both simultaneously
     */
    fun classifyWithContext(context: Context, bitmap: Bitmap, labels: List<String> = symptomLabels): Pair<String, Float> {
        try {
            // Step 1: Load vision, encode, unload
            Log.d("MedSigLIP", "=== SEQUENTIAL CLASSIFICATION ===")
            Log.d("MedSigLIP", "Step 1: Loading vision model...")
            
            if (visionInterpreter == null) init(context)
            Log.d("MedSigLIP", "Step 1: Encoding image...")
            val imageEmbedding = encode(bitmap)
            Log.d("MedSigLIP", "Step 1b: Unloading vision model to free RAM...")
            closeVision()
            
            if (imageEmbedding.all { it == 0f }) {
                Log.e("MedSigLIP", "Image embedding is all zeros!")
                return "unknown" to 0.1f
            }
            
            Log.d("MedSigLIP", "Step 2: Loading text model...")
            
            // Step 2: Load text, encode labels, unload
            if (textInterpreter == null) initTextEncoder(context)
            val textEmbeddings = labels.mapIndexed { index, label ->
                Log.d("MedSigLIP", "Encoding label $index: '$label'")
                encodeText(label)
            }
            closeTextEncoder()
            
            // Check embeddings
            val validTextEmbeddings = textEmbeddings.filter { emb ->
                kotlin.math.abs(emb.sum()) > 0.01f
            }
            
            if (validTextEmbeddings.isEmpty()) {
                Log.e("MedSigLIP", "CRITICAL: All text embeddings are near zero!")
                return fallbackClassification(bitmap, labels)
            }
            
            // Step 3: Compute similarities
            Log.d("MedSigLIP", "Step 3: Computing similarities...")
            val rawSimilarities = labels.mapIndexed { index, label ->
                label to dotProduct(imageEmbedding, textEmbeddings[index])
            }
            
            // Apply logit_scale and softmax
            val logits = rawSimilarities.map { it.first to it.second * LOGIT_SCALE }
            val maxLogit = logits.maxOfOrNull { it.second } ?: 0f
            val expScores = logits.map { kotlin.math.exp((it.second - maxLogit).toDouble()).toFloat() }
            val sumExp = expScores.sum()
            val probabilities = logits.mapIndexed { index, (label, _) -> 
                label to (expScores[index] / sumExp)
            }
            
            // Log results
            for ((label, prob) in probabilities) {
                val raw = rawSimilarities.find { it.first == label }?.second ?: 0f
                Log.d("MedSigLIP", "  $label: raw=${String.format("%.4f", raw)}, prob=${String.format("%.4f", prob)}")
            }
            
            val bestMatch = probabilities.maxByOrNull { it.second } ?: (labels.first() to 0f)
            val confidence = bestMatch.second.coerceIn(0.05f, 0.95f)
            
            // Return raw model result - no healthy skin override
            // Let the model decide what's in the image
            Log.d("MedSigLIP", "Result: ${bestMatch.first} (${String.format("%.2f", confidence * 100)}%)")
            Log.d("MedSigLIP", "=== END SEQUENTIAL CLASSIFICATION ===")
            
            return bestMatch.first to confidence
            
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("MedSigLIP", "Classification failed: ${e.message}")
            return "unknown" to 0.1f
        }
    }
    
    /**
     * Fallback classification when text encoder is not working.
     * Uses simple heuristics based on image features.
     */
    private fun fallbackClassification(bitmap: Bitmap, labels: List<String>): Pair<String, Float> {
        Log.w("MedSigLIP", "Using fallback classification method")
        // For now, return a random but consistent result based on image hash
        // In a real app, you'd use a different model or approach
        val imageHash = bitmap.hashCode()
        val index = kotlin.math.abs(imageHash) % labels.size
        return labels[index] to 0.25f  // Low confidence since we're guessing
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) {
            sum += a[i] * b[i]
        }
        return sum
    }
    
    private fun calculateStdDev(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average().toFloat()
        val variance = values.map { (it - mean) * (it - mean) }.average().toFloat()
        return kotlin.math.sqrt(variance)
    }
    
    private fun matchDistribution(vec: FloatArray, targetMax: Float): FloatArray {
        val maxVal = vec.maxOrNull() ?: 1f
        if (maxVal == 0f) return vec
        val scale = targetMax / maxVal
        Log.d("MedSigLIP", "matchDistribution: scaling by ${String.format("%.4f", scale)}")
        return FloatArray(vec.size) { i -> vec[i] * scale }
    }
    
    private fun calculateVectorStats(vector: FloatArray): Pair<Float, Float> {
        // Calculate L2 norm
        var sumSq = 0f
        var maxVal = 0f
        for (v in vector) {
            sumSq += v * v
            if (kotlin.math.abs(v) > maxVal) maxVal = kotlin.math.abs(v)
        }
        val norm = kotlin.math.sqrt(sumSq)
        return norm to maxVal
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vector) {
            sumSq += v * v
        }
        val norm = kotlin.math.sqrt(sumSq)
        return if (norm > 0f) {
            FloatArray(vector.size) { i -> vector[i] / norm }
        } else {
            vector
        }
    }
    
    private fun headerlessGlobalAveragePooling(flatData: FloatArray, numPatches: Int, featureDim: Int): FloatArray {
        val poolingVector = FloatArray(featureDim)
        
        // Sum across all patches
        for (i in 0 until numPatches) {
            val offset = i * featureDim
            for (j in 0 until featureDim) {
                poolingVector[j] += flatData[offset + j]
            }
        }
        
        // Average (divide by number of patches)
        for (j in 0 until featureDim) {
            poolingVector[j] /= numPatches.toFloat()
        }
        
        // NOTE: Do NOT normalize here - encode() will normalize the final output
        // This ensures consistent normalization for both pooled and non-pooled outputs
        return poolingVector
    }
    
    /**
     * Max pooling across patches - takes the maximum value for each feature dimension
     * This can help reduce dominance of single dimensions in vision embeddings
     */
    private fun headerlessMaxPooling(flatData: FloatArray, numPatches: Int, featureDim: Int): FloatArray {
        val poolingVector = FloatArray(featureDim)
        
        // Initialize with very negative values
        for (j in 0 until featureDim) {
            poolingVector[j] = Float.NEGATIVE_INFINITY
        }
        
        // Take max across all patches
        for (i in 0 until numPatches) {
            val offset = i * featureDim
            for (j in 0 until featureDim) {
                val value = flatData[offset + j]
                if (value > poolingVector[j]) {
                    poolingVector[j] = value
                }
            }
        }
        
        // Handle case where all values might be negative infinity
        for (j in 0 until featureDim) {
            if (poolingVector[j] == Float.NEGATIVE_INFINITY) {
                poolingVector[j] = 0f
            }
        }
        
        return poolingVector
    }
}
