const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const app = express();
const PORT = 3000;

app.use(express.json());

const GEMINI_API_KEY = 'AIzaSyAaK_g64tXlaXIhFzxj-eUT9RzprG7n3xM'

const GEMINI_API_URL_CONTENT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;
const GEMINI_API_URL_EMBEDDING = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${GEMINI_API_KEY}`;
function chunkText(text, maxTokens = 500, overlapTokens = 100) {
    const chunks = [];
    // Split by paragraphs first (one or more newlines with optional whitespace)
    const paragraphs = text.split(/\n\s*\n/);

    for (const para of paragraphs) {
        if (para.trim() === '') continue; // Skip empty paragraphs

        const words = para.split(/\s+/); // Split paragraph into words
        let currentChunkWords = [];

        for (let i = 0; i < words.length; i++) {
            currentChunkWords.push(words[i]);
            // If current chunk reaches maxTokens, create a chunk and reset for the next
            if (currentChunkWords.length >= maxTokens) {
                chunks.push(currentChunkWords.join(' '));
                // For overlap, slice from the end of the current chunk
                currentChunkWords = currentChunkWords.slice(Math.max(0, currentChunkWords.length - overlapTokens));
            }
        }
        // Add any remaining words from the paragraph as the last chunk
        if (currentChunkWords.length > 0) {
            chunks.push(currentChunkWords.join(' '));
        }
    }
    return chunks;
}

/**
 * Calls the Gemini Embedding API to get a vector embedding for a given text.
 * @param {string} text - The text to embed.
 * @returns {Promise<number[]>} A promise that resolves to the embedding vector.
 * @throws {Error} If the API call fails.
 */
async function getEmbedding(text) {
    try {
        const response = await axios.post(
            GEMINI_API_URL_EMBEDDING,
            {
                // Specify the embedding model
                model: "text-embedding-004",
                content: { parts: [{ text: text }] }
            },
            {
                headers: { 'Content-Type': 'application/json' },
                timeout: 10000 // Add a timeout for embedding calls
            }
        );
        if (!response.data || !response.data.embedding || !response.data.embedding.values) {
            throw new Error("Invalid embedding response from Gemini.");
        }
        return response.data.embedding.values;
    } catch (error) {
        console.error("Error getting embedding:", error.response ? error.response.data : error.message);
        throw new Error("Failed to get embedding for text.");
    }
}

/**
 * Calculates the cosine similarity between two vectors.
 * Used to find the most relevant text chunks.
 * @param {number[]} vecA - The first vector.
 * @param {number[]} vecB - The second vector.
 * @returns {number} The cosine similarity score (between -1 and 1).
 */
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
    if (magnitudeA === 0 || magnitudeB === 0) return 0; // Avoid division by zero
    return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Converts common number words (e.g., "thirty-six") to digits (e.g., "36").
 * This is a fallback for LLM output, as strong prompting should make it less necessary.
 * @param {string} text - The input text possibly containing number words.
 * @returns {string} The text with number words converted to digits.
 */
function convertWordsToDigits(text) {
    const wordToDigitMap = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000',
        'thirty-six': '36', 'twenty-four': '24', // Add more compound numbers as needed
        'two': '2' // Ensure specific numbers like 'two' are covered
    };

    let convertedText = text;
    for (const word in wordToDigitMap) {
        // Use regex with word boundaries to avoid partial matches (e.g., "one" in "money")
        const regex = new RegExp(`\\b${word}\\b`, 'gi');
        convertedText = convertedText.replace(regex, wordToDigitMap[word]);
    }
    return convertedText;
}


// --- API Endpoint ---

/**
 * Single endpoint to handle both PDF processing and question answering.
 * Note: This will re-process the PDF on every request, impacting performance and cost.
 */
app.post('/hackrx/run', async (req, res) => {
    const startTime = process.hrtime.bigint(); // Start timing the request

    try {
        const { documents, questions } = req.body; // Now expects both documents (URL) and questions

        if (!documents || !Array.isArray(questions) || questions.length === 0) {
            return res.status(400).json({ error: 'Invalid input format. "documents" (URL) and "questions" (array of strings) are required.' });
        }
        if (!GEMINI_API_KEY) {
            return res.status(500).json({ error: 'API key for Gemini is not configured. Please set GEMINI_API_KEY in your .env file.' });
        }

        // --- Phase 1: PDF Processing and Indexing (Happens on every request now) ---
        console.log(`Processing PDF from: ${documents}`);
        const pdfResponse = await axios.get(documents, { responseType: 'arraybuffer', timeout: 90000 }); // Increased timeout
        const pdfData = await pdfParse(pdfResponse.data);
        const extractedText = pdfData.text;
        console.log('Extracted Text Length:', extractedText.length);

        const chunks = chunkText(extractedText, 500, 100); // Adjust maxTokens and overlapTokens as needed
        console.log(`Generated ${chunks.length} chunks.`);

        const documentChunks = []; // Local to this request
        // Generate embeddings for each chunk in parallel
        await Promise.all(chunks.map(async (chunk, index) => {
            try {
                const embedding = await getEmbedding(chunk);
                documentChunks.push({ text: chunk, embedding: embedding });
                // console.log(`Embedded chunk ${index + 1}/${chunks.length}`); // Too verbose for every request
            } catch (error) {
                console.error(`Failed to embed chunk ${index + 1}:`, error.message);
            }
        }));
        console.log(`Finished embedding ${documentChunks.length} chunks for this request.`);
        // --- End Phase 1 ---

        // --- Phase 2: Question Answering ---
        const answerPromises = questions.map(async (question) => {
            // 1. Get embedding for the current question
            const questionEmbedding = await getEmbedding(question);

            // 2. Retrieve top N relevant chunks from our indexed document
            const similarities = documentChunks.map(chunk => ({
                chunk: chunk.text,
                similarity: cosineSimilarity(questionEmbedding, chunk.embedding)
            }));

            // Sort by similarity in descending order
            similarities.sort((a, b) => b.similarity - a.similarity);

            // Select the top N most relevant chunks
            const topN = 7; // Increased topN for better coverage
            const relevantContextChunks = similarities.slice(0, topN).map(s => s.chunk);

            // Combine relevant chunks into a single context string for the LLM
            const context = relevantContextChunks.join('\n\n');

            // Define a maximum context length to prevent exceeding LLM token limits
            const MAX_LLM_CONTEXT_LENGTH = 30000; // Adjust based on Gemini 1.5 Flash's actual context window
            const finalContext = context.length > MAX_LLM_CONTEXT_LENGTH ? context.substring(0, MAX_LLM_CONTEXT_LENGTH) : context;

            // 3. Construct the prompt for Gemini (for each specific question)
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

            // 4. Make the API call to Google Gemini for this specific question
            try {
                const geminiResponse = await axios.post(
                    GEMINI_API_URL_CONTENT,
                    prompt,
                    {
                        headers: { 'Content-Type': 'application/json' },
                        timeout: 30000 // Timeout for the Gemini content generation call
                    }
                );
                const output = geminiResponse.data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response from Gemini.';

                // Post-process the output for cleaning and number conversion
                let cleanedAnswer = output.trim();
                cleanedAnswer = cleanedAnswer
                    .replace(/^\d+\.\s*/, '')      // Remove leading numbers like "1. "
                    .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown characters
                    .replace(/\(\d+\.\d+\)/g, '')     // Remove parenthesized numbers (e.g., page numbers)
                    .replace(/^- /, '')              // Remove leading hyphens
                    // Aggressive removal of common LLM intro/outro phrases
                    .replace(/The answer is: /i, '')
                    .replace(/Based on the document, /i, '')
                    .replace(/According to the policy, /i, '')
                    .replace(/It states that /i, '')
                    .replace(/Here is the answer: /i, '')
                    .replace(/The document states: /i, '')
                    .replace(/Answer: /i, '') // Added this
                    .replace(/The policy states: /i, '') // Added this
                    .trim();

                // Attempt to convert number words to digits as a fallback
                cleanedAnswer = convertWordsToDigits(cleanedAnswer);

                return cleanedAnswer;
            } catch (geminiError) {
                console.error(`Error asking Gemini for question "${question}":`, geminiError.response ? geminiError.response.data : geminiError.message);
                // Return "Not found in document." if there's an API error
                return "Not found in document.";
            }
        });

        const answers = await Promise.all(answerPromises); // Wait for all questions to be answered

        const endTime = process.hrtime.bigint(); // End timing
        const responseTimeMs = Number(endTime - startTime) / 1_000_000; // Convert to milliseconds

        res.json({ answers, responseTimeMs: `${responseTimeMs.toFixed(2)} ms` });

    } catch (err) {
        console.error('Error processing request:', err.message);
        const endTime = process.hrtime.bigint(); // End timing even on error
        const responseTimeMs = Number(endTime - startTime) / 1_000_000; // Convert to milliseconds
        res.status(500).json({ error: 'Something went wrong', detail: err.message, responseTimeMs: `${responseTimeMs.toFixed(2)} ms` });
    }
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});
