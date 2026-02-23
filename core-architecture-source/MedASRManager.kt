package com.medgemma.forensic.data

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineMedAsrCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Manages MedASR (Sherpa ONNX) for transcription.
 * Uses the official sherpa-onnx v1.12.24 AAR (metadata-stripped) or source integration.
 */
object MedASRManager {

    private var recognizer: OfflineRecognizer? = null

    suspend fun init(context: Context) = withContext(Dispatchers.IO) {
        if (recognizer != null) return@withContext

        val modelFile = ModelDownloader.getModelFile(context, ModelDownloader.ModelType.EAR)
        val tokensFile = ModelDownloader.getModelFile(context, ModelDownloader.ModelType.EAR_TOKENS)

        if (!modelFile.exists() || !tokensFile.exists() || modelFile.length() == 0L || tokensFile.length() == 0L) {
            val msg = "Model or tokens missing/empty. Initialization aborted."
            Log.w("MedASR", msg)
            throw RuntimeException(msg)
        }

        try {
            Log.d("MedASR", "Loading model: ${modelFile.absolutePath} (size=${modelFile.length()} bytes)")
            Log.d("MedASR", "Loading tokens: ${tokensFile.absolutePath} (size=${tokensFile.length()} bytes)")

            val medAsrConfig = OfflineMedAsrCtcModelConfig(
                model = modelFile.absolutePath
            )
            Log.d("MedASR", "Created medAsrConfig. model='${medAsrConfig.model}'")

            val modelConfig = OfflineModelConfig(
                medasr = medAsrConfig,
                tokens = tokensFile.absolutePath,
                numThreads = 1, // Reduced to 1 for safety on older devices
                debug = false, // Disable debug to reduce log noise
                provider = "cpu",
                // Explicitly set modelType for MedASR
                modelType = "medasr"
            )
            Log.d("MedASR", "OfflineModelConfig built. medasr.model='${modelConfig.medasr.model}'")

            val featConfig = FeatureConfig(
                sampleRate = 16000,
                featureDim = 80
            )

            val recognizerConfig = OfflineRecognizerConfig(
                featConfig = featConfig,
                modelConfig = modelConfig
            )

            // Initialize recognizer
            recognizer = OfflineRecognizer(
                assetManager = null, 
                config = recognizerConfig
            )
            Log.d("MedASR", "MedASR initialized successfully")

        } catch (e: UnsatisfiedLinkError) {
            val msg = "Native library failed to load: ${e.message}"
            Log.e("MedASR", msg)
            throw RuntimeException(msg, e)
        } catch (e: Exception) {
            val msg = "Initialization failed: ${e.message}"
            Log.e("MedASR", msg)
            e.printStackTrace()
            throw RuntimeException(msg, e)
        }
    }

    fun close() {
        recognizer?.release()
        recognizer = null
    }

    /**
     * Load audio file and convert to FloatArray at 16kHz for MedASR
     * Supports: M4A, WAV, MP3, OGG, and other common audio formats
     */
    suspend fun loadAudioFile(audioFile: File): FloatArray = withContext(Dispatchers.IO) {
        try {
            Log.d("MedASR", "Loading audio file: ${audioFile.absolutePath}")
            
            val extractor = MediaExtractor()
            extractor.setDataSource(audioFile.absolutePath)
            
            // Find audio track
            var audioTrackIndex = -1
            var audioFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME)
                if (mime?.startsWith("audio/") == true) {
                    audioTrackIndex = i
                    audioFormat = format
                    break
                }
            }
            
            if (audioTrackIndex == -1 || audioFormat == null) {
                throw RuntimeException("No audio track found in file")
            }
            
            val originalSampleRate = audioFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = audioFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            
            Log.d("MedASR", "Original: ${originalSampleRate}Hz, ${channelCount} channels")
            
            extractor.selectTrack(audioTrackIndex)
            
            val decoder = MediaCodec.createDecoderByType(
                audioFormat.getString(MediaFormat.KEY_MIME) ?: "audio/mp4a-latm"
            )
            decoder.configure(audioFormat, null, null, 0)
            decoder.start()
            
            val outputFormat = decoder.outputFormat
            val outputSampleRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val outputChannels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            
            Log.d("MedASR", "Decoded: ${outputSampleRate}Hz, ${outputChannels} channels")
            
            // Collect decoded PCM data
            val pcmData = mutableListOf<Short>()
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            
            while (!outputDone) {
                // Feed input
                if (!inputDone) {
                    val inputBufferIndex = decoder.dequeueInputBuffer(10000)
                    if (inputBufferIndex >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputBufferIndex)
                        if (inputBuffer != null) {
                            val sampleSize = extractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                decoder.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                decoder.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }
                }
                
                // Get output
                val outputBufferIndex = decoder.dequeueOutputBuffer(bufferInfo, 10000)
                when {
                    outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        // Format changed
                    }
                    outputBufferIndex >= 0 -> {
                        val outputBuffer = decoder.getOutputBuffer(outputBufferIndex)
                        if (outputBuffer != null && bufferInfo.size > 0) {
                            outputBuffer.position(bufferInfo.offset)
                            outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                            
                            // Convert to short array (16-bit PCM)
                            val shortBuffer = outputBuffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                            while (shortBuffer.hasRemaining()) {
                                pcmData.add(shortBuffer.get())
                            }
                        }
                        decoder.releaseOutputBuffer(outputBufferIndex, false)
                        
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputDone = true
                        }
                    }
                }
            }
            
            decoder.stop()
            decoder.release()
            extractor.release()
            
            // Convert to FloatArray and resample to 16kHz if needed
            val pcmShortArray = pcmData.toShortArray()
            val resampled = if (originalSampleRate != 16000) {
                resampleTo16kHz(pcmShortArray, originalSampleRate)
            } else {
                pcmShortArray
            }
            
            // Convert short to float (normalized to -1.0 to 1.0)
            val floatArray = FloatArray(resampled.size)
            for (i in resampled.indices) {
                floatArray[i] = resampled[i].toFloat() / 32768.0f
            }
            
            Log.d("MedASR", "Audio loaded: ${floatArray.size} samples at 16kHz")
            floatArray
            
        } catch (e: Exception) {
            Log.e("MedASR", "Error loading audio: ${e.message}")
            throw RuntimeException("Failed to load audio: ${e.message}", e)
        }
    }
    
    /**
     * Simple linear resampling to 16kHz
     */
    private fun resampleTo16kHz(data: ShortArray, originalRate: Int): ShortArray {
        val ratio = originalRate.toDouble() / 16000.0
        val newLength = (data.size / ratio).toInt()
        val result = ShortArray(newLength)
        
        for (i in 0 until newLength) {
            val srcIndex = (i * ratio).toInt().coerceIn(0, data.size - 1)
            result[i] = data[srcIndex]
        }
        
        return result
    }

    /**
     * Transcribe audio file using MedASR
     */
    suspend fun transcribeAudioFile(audioFile: File): String = withContext(Dispatchers.Default) {
        if (recognizer == null) {
            return@withContext "Error: MedASR not initialized"
        }
        
        try {
            Log.d("MedASR", "Starting transcription for: ${audioFile.name}")
            
            // Load and convert audio to FloatArray
            val samples = loadAudioFile(audioFile)
            
            // Transcribe
            transcribe(samples)
        } catch (e: Exception) {
            Log.e("MedASR", "Transcription error: ${e.message}")
            "Error transcribing audio: ${e.message}"
        }
    }

    /**
     * Transcribe 16kHz audio samples (FloatArray)
     */
    suspend fun transcribe(samples: FloatArray): String = withContext(Dispatchers.Default) {
        val r = recognizer ?: return@withContext "Error: MedASR not initialized"
        try {
            val stream = r.createStream()
            stream.acceptWaveform(samples, 16000)
            r.decode(stream)
            val result = r.getResult(stream)
            val text = result.text
            stream.release()
            
            if (text.isEmpty()) "(No speech detected)" else text
        } catch (e: Exception) {
            "Error transcribing: ${e.message}"
        }
    }
}