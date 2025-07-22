const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const ort = require('onnxruntime-node'); // Import onnxruntime-node
const { AutoTokenizer } = require('@huggingface/tokenizers'); // Import tokenizer

const app = express();
const PORT = 3000;

app.use(express.json());

// --- Configuration ---
const GEMINI_API_KEY = 'AIzaSyC3loxj3YwITkUl6gDK3QY53QYvHElLhrw'; // Ensure this is set in your .env file

const GEMINI_API_URL_CONTENT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;

let localEmbeddingSession = null;
let localTokenizer = null;
const LOCAL_MODEL_PATH = './models/model.onnx'; // Adjust if your model file name is different (e.g., all-MiniLM-L6-v2.onnx)
const LOCAL_TOKENIZER_PATH = './models/tokenizer.json'; // Adjust if your tokenizer file name is different

/**
 * Function to initialize the local embedding model and tokenizer.
 * This runs once when the server starts.
 */
async function initializeLocalEmbeddingModel() {
    try {
        console.log("Initializing local embedding model...");
        localEmbeddingSession = await ort.InferenceSession.create(LOCAL_MODEL_PATH);
        localTokenizer = await AutoTokenizer.from_file(LOCAL_TOKENIZER_PATH);
        console.log("Local embedding model and tokenizer initialized successfully.");
    } catch (error) {
        console.error("Failed to initialize local embedding model:", error);
        console.error("Please ensure 'onnxruntime-node' and '@huggingface/tokenizers' are installed,");
        console.error("and that the ONNX model and tokenizer files exist at the specified paths:");
        console.error(`Model Path: ${LOCAL_MODEL_PATH}`);
        console.error(`Tokenizer Path: ${LOCAL_TOKENIZER_PATH}`);
        // If local model fails, you might want to throw an error or fall back to Gemini API embedding here.
        // For now, the app will continue but getEmbedding will throw if it's not initialized.
    }
}

// Call this initialization function when your app starts
initializeLocalEmbeddingModel();

// --- Utility Functions (chunkText, cosineSimilarity, convertWordsToDigits) ---
// These functions remain the same as in the previous code block.

function chunkText(text, maxTokens = 500, overlapTokens = 100) {
    const chunks = [];
    const paragraphs = text.split(/\n\s*\n/);

    for (const para of paragraphs) {
        if (para.trim() === '') continue;

        const words = para.split(/\s+/);
        let currentChunkWords = [];

        for (let i = 0; i < words.length; i++) {
            currentChunkWords.push(words[i]);
            if (currentChunkWords.length >= maxTokens) {
                chunks.push(currentChunkWords.join(' '));
                currentChunkWords = currentChunkWords.slice(Math.max(0, currentChunkWords.length - overlapTokens));
            }
        }
        if (currentChunkWords.length > 0) {
            chunks.push(currentChunkWords.join(' '));
        }
    }
    return chunks;
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] * vecA[i];
        magnitudeB += vecB[i] * vecB[i];
    }
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
}

function convertWordsToDigits(text) {
    const wordToDigitMap = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000',
        'thirty-six': '36', 'twenty-four': '24',
        'two': '2'
    };

    let convertedText = text;
    for (const word in wordToDigitMap) {
        const regex = new RegExp(`\\b${word}\\b`, 'gi');
        convertedText = convertedText.replace(regex, wordToDigitMap[word]);
    }
    return convertedText;
}


/**
 * Gets embeddings using the locally loaded ONNX model.
 * @param {string} text - The text to embed.
 * @returns {Promise<number[]>} A promise that resolves to the embedding vector.
 * @throws {Error} If the local model is not initialized or inference fails.
 */
async function getEmbedding(text) {
    if (!localEmbeddingSession || !localTokenizer) {
        throw new Error("Local embedding model is not initialized. Cannot perform local embedding.");
    }

    try {
        // 1. Tokenize the input text
        const encoding = await localTokenizer.encode(text);

        // Create ONNX Runtime Tensors from tokenized output
        // Ensure BigInt64Array is used for int64 tensors
        const input_ids = new ort.Tensor('int64', BigInt64Array.from(encoding.ids.map(id => BigInt(id))), [1, encoding.ids.length]);
        const attention_mask = new ort.Tensor('int64', BigInt64Array.from(encoding.attention_mask.map(id => BigInt(id))), [1, encoding.attention_mask.length]);
        // token_type_ids might not be present in all models, check your model's inputs
        const token_type_ids = new ort.Tensor('int64', BigInt64Array.from(encoding.type_ids.map(id => BigInt(id))), [1, encoding.type_ids.length]);


        // 2. Define model inputs (adjust input names if your model uses different ones)
        const feeds = {
            input_ids: input_ids,
            attention_mask: attention_mask,
            token_type_ids: token_type_ids // Remove if your model doesn't use token_type_ids
        };

        // 3. Run inference
        const results = await localEmbeddingSession.run(feeds);

        // 4. Extract embeddings (output name might vary, common is 'last_hidden_state' or 'pooler_output')
        // For sentence-transformers, often the output is 'last_hidden_state' and you need to pool it.
        // This is a common pooling strategy: mean pooling over the last_hidden_state.
        // Assuming 'last_hidden_state' is the output name and it's a Float32Array
        const lastHiddenState = results.last_hidden_state.data; // Adjust output name if different
        const inputMask = encoding.attention_mask; // Use attention mask for pooling

        const embeddings = new Array(results.last_hidden_state.dims[2]).fill(0); // Initialize with zeros
        let sum = 0;

        for (let i = 0; i < inputMask.length; i++) {
            if (inputMask[i] === 1) { // Only consider non-padded tokens
                for (let j = 0; j < embeddings.length; j++) {
                    embeddings[j] += lastHiddenState[i * embeddings.length + j];
                }
                sum++;
            }
        }

        if (sum > 0) {
            for (let i = 0; i < embeddings.length; i++) {
                embeddings[i] /= sum; // Mean pooling
            }
        }

        return embeddings; // Return the pooled embedding array

    } catch (error) {
        console.error("Error during local embedding inference:", error);
        throw new Error("Local embedding failed. See server logs for details.");
    }
}


// --- API Endpoint ---

app.post('/hackrx/run', async (req, res) => {
    const startTime = process.hrtime.bigint();

    try {
        const { documents, questions } = req.body;

        if (!documents || !Array.isArray(questions) || questions.length === 0) {
            return res.status(400).json({ error: 'Invalid input format. "documents" (URL) and "questions" (array of strings) are required.' });
        }
        if (!GEMINI_API_KEY) {
            return res.status(500).json({ error: 'API key for Gemini is not configured. Please set GEMINI_API_KEY in your .env file.' });
        }

        // --- Phase 1: PDF Processing and Indexing (Happens on every request now) ---
        console.log(`Processing PDF from: ${documents}`);
        const pdfResponse = await axios.get(documents, { responseType: 'arraybuffer', timeout: 90000 });
        const pdfData = await pdfParse(pdfResponse.data);
        const extractedText = pdfData.text;
        console.log('Extracted Text Length:', extractedText.length);

        const chunks = chunkText(extractedText, 500, 100);
        console.log(`Generated ${chunks.length} chunks.`);

        const documentChunks = [];
        await Promise.all(chunks.map(async (chunk, index) => {
            try {
                const embedding = await getEmbedding(chunk); // Now uses local embedding
                documentChunks.push({ text: chunk, embedding: embedding });
            } catch (error) {
                console.error(`Failed to embed chunk ${index + 1}:`, error.message);
            }
        }));
        console.log(`Finished embedding ${documentChunks.length} chunks for this request.`);
        // --- End Phase 1 ---

        // --- Phase 2: Question Answering ---
        const answerPromises = questions.map(async (question) => {
            const questionEmbedding = await getEmbedding(question); // Now uses local embedding

            const similarities = documentChunks.map(chunk => ({
                chunk: chunk.text,
                similarity: cosineSimilarity(questionEmbedding, chunk.embedding)
            }));

            similarities.sort((a, b) => b.similarity - a.similarity);
            const topN = 7;
            const relevantContextChunks = similarities.slice(0, topN).map(s => s.chunk);
            const context = relevantContextChunks.join('\n\n');
            const MAX_LLM_CONTEXT_LENGTH = 30000;
            const finalContext = context.length > MAX_LLM_CONTEXT_LENGTH ? context.substring(0, MAX_LLM_CONTEXT_LENGTH) : context;

            const prompt = {
                contents: [
                    {
                        role: 'user',
                        parts: [
                            {
                                text: `Context:\n${finalContext}\n\nAnswer the following question based ONLY on the provided document context. Your answer MUST be:
1.  Extremely concise and direct.
2.  Contain ONLY the requested information.
3.  Use digits for all numbers (e.g., "36" not "thirty-six").
4.  Do NOT include any introductory phrases, explanations, or concluding remarks.
5.  If the answer is a definition, provide the exact definition as found in the text.
6.  If the answer is not explicitly found in the document context, state "Not found in document."
7.  Do NOT add any information that is not directly present in the provided context.

Question: ${question}`
                            }
                        ]
                    }
                ]
            };

            try {
                const geminiResponse = await axios.post(
                    GEMINI_API_URL_CONTENT,
                    prompt,
                    {
                        headers: { 'Content-Type': 'application/json' },
                        timeout: 30000
                    }
                );
                const output = geminiResponse.data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response from Gemini.';

                let cleanedAnswer = output.trim();
                cleanedAnswer = cleanedAnswer
                    .replace(/^\d+\.\s*/, '')
                    .replace(/\*\*(.*?)\*\*/g, '$1')
                    .replace(/\(\d+\.\d+\)/g, '')
                    .replace(/^- /, '')
                    .replace(/The answer is: /i, '')
                    .replace(/Based on the document, /i, '')
                    .replace(/According to the policy, /i, '')
                    .replace(/It states that /i, '')
                    .replace(/Here is the answer: /i, '')
                    .replace(/The document states: /i, '')
                    .replace(/Answer: /i, '')
                    .replace(/The policy states: /i, '')
                    .trim();

                cleanedAnswer = convertWordsToDigits(cleanedAnswer);

                return cleanedAnswer;
            } catch (geminiError) {
                console.error(`Error asking Gemini for question "${question}":`, geminiError.response ? geminiError.response.data : geminiError.message);
                return "Not found in document.";
            }
        });

        const answers = await Promise.all(answerPromises);
    } catch (err) {
        console.error('Error processing request:', err.message);
        const endTime = process.hrtime.bigint();
        const responseTimeMs = Number(endTime - startTime) / 1_000_000;
        res.status(500).json({ error: 'Something went wrong', detail: err.message, responseTimeMs: `${responseTimeMs.toFixed(2)} ms` });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});
