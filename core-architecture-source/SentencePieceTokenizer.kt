package com.medgemma.forensic.ai

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.ln

/**
 * SentencePiece Unigram Tokenizer for MedSigLIP
 * 
 * Implements the Unigram algorithm used by SentencePiece to properly tokenize
 * medical text with the full 32k vocabulary from tokenizer.json.
 */
class SentencePieceTokenizer(context: Context) {
    
    companion object {
        private const val TAG = "SentencePieceTokenizer"
        
        // Special token IDs from MedSigLIP
        const val PAD_TOKEN_ID = 0
        const val EOS_TOKEN_ID = 1
        const val UNK_TOKEN_ID = 2
        
        const val PAD_TOKEN = "<pad>"
        const val EOS_TOKEN = "</s>"
        const val UNK_TOKEN = "<unk>"
    }
    
    private data class TokenScore(val token: String, val score: Double, val id: Int)
    
    private val vocab = mutableListOf<TokenScore>()
    private val tokenToId = mutableMapOf<String, Int>()
    private var isLoaded = false
    
    init {
        try {
            Log.d(TAG, "Loading MedSigLIP Unigram tokenizer...")
            loadTokenizer(context)
            
            if (vocab.size < 1000) {
                Log.e(TAG, "ERROR: Only ${vocab.size} tokens loaded!")
                isLoaded = false
            } else {
                isLoaded = true
                Log.d(TAG, "✅ Loaded ${vocab.size} tokens")
                Log.d(TAG, "  PAD: '$PAD_TOKEN' -> $PAD_TOKEN_ID")
                Log.d(TAG, "  EOS: '$EOS_TOKEN' -> $EOS_TOKEN_ID")
                Log.d(TAG, "  UNK: '$UNK_TOKEN' -> $UNK_TOKEN_ID")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load tokenizer", e)
            isLoaded = false
        }
    }
    
    /**
     * Load vocabulary from tokenizer.json
     */
    private fun loadTokenizer(context: Context) {
        val jsonText = context.assets.open("tokenizer.json").bufferedReader().use { it.readText() }
        val root = JSONObject(jsonText)
        val model = root.getJSONObject("model")
        val vocabArray = model.getJSONArray("vocab")
        
        Log.d(TAG, "Model type: ${model.getString("type")}")
        Log.d(TAG, "Vocab entries: ${vocabArray.length()}")
        
        // Load all tokens with their scores
        for (i in 0 until vocabArray.length()) {
            val entry = vocabArray.getJSONArray(i)
            val token = entry.getString(0)
            val score = entry.getDouble(1)
            
            vocab.add(TokenScore(token, score, i))
            tokenToId[token] = i
        }
        
        // Sort by token length (longest first) for efficient matching
        vocab.sortByDescending { it.token.length }
    }
    
    /**
     * Tokenize text using Unigram algorithm
     * 
     * This implements a simplified version of the SentencePiece Unigram algorithm:
     * 1. Normalize text (lowercase, add space prefix)
     * 2. Find all possible tokenizations using dynamic programming
     * 3. Select the tokenization with highest probability (sum of log scores)
     */
    fun encode(text: String): List<Int> {
        if (!isLoaded) {
            Log.e(TAG, "Tokenizer not loaded!")
            return listOf(EOS_TOKEN_ID, UNK_TOKEN_ID, EOS_TOKEN_ID)
        }
        
        // Normalize: add space prefix (SentencePiece convention) and lowercase
        val normalized = "▁" + text.lowercase().replace(" ", "▁")
        
        // Tokenize using Viterbi algorithm (dynamic programming)
        val tokens = tokenizeUnigram(normalized)
        
        // Add EOS tokens at start and end
        val result = mutableListOf(EOS_TOKEN_ID)
        result.addAll(tokens)
        result.add(EOS_TOKEN_ID)
        
        Log.d(TAG, "Tokenized '$text' -> $result")
        
        return result
    }
    
    /**
     * Unigram tokenization using Viterbi algorithm
     * 
     * Finds the best tokenization by maximizing the sum of token scores.
     */
    private fun tokenizeUnigram(text: String): List<Int> {
        val n = text.length
        if (n == 0) return emptyList()
        
        // DP arrays: best[i] = best score for text[0..i), backtrack[i] = token end position
        val best = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val backtrack = IntArray(n + 1) { -1 }
        best[0] = 0.0
        
        // Forward pass: find best tokenization
        for (i in 0 until n) {
            if (best[i] == Double.NEGATIVE_INFINITY) continue
            
            // Try all possible tokens starting at position i
            for (tokenScore in vocab) {
                val token = tokenScore.token
                val end = i + token.length
                
                if (end <= n && text.substring(i, end) == token) {
                    val score = best[i] + tokenScore.score
                    if (score > best[end]) {
                        best[end] = score
                        backtrack[end] = i
                    }
                }
            }
            
            // Fallback: single character as UNK
            if (best[i + 1] == Double.NEGATIVE_INFINITY) {
                best[i + 1] = best[i] - 10.0  // Penalty for UNK
                backtrack[i + 1] = i
            }
        }
        
        // Backward pass: reconstruct best tokenization
        val result = mutableListOf<Int>()
        var pos = n
        
        while (pos > 0) {
            val start = backtrack[pos]
            if (start < 0) break
            
            val substring = text.substring(start, pos)
            val tokenId = tokenToId[substring] ?: UNK_TOKEN_ID
            result.add(0, tokenId)  // Add to front
            
            pos = start
        }
        
        return result
    }
    
    /**
     * Decode token IDs back to text
     */
    fun decode(tokenIds: List<Int>): String {
        return tokenIds
            .filter { it != PAD_TOKEN_ID && it != EOS_TOKEN_ID }
            .mapNotNull { id -> vocab.find { it.id == id }?.token }
            .joinToString("")
            .replace("▁", " ")
            .trim()
    }
    
    fun isReady(): Boolean = isLoaded
    fun getVocabSize(): Int = vocab.size
}
