const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');

const app = express();
const PORT = 3000;

app.use(express.json());

// --- Configuration ---
const GEMINI_API_KEY = 'AIzaSyAaK_g64tXlaXIhFzxj-eUT9RzprG7n3xM';
const GEMINI_API_URL_CONTENT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;
const GEMINI_API_URL_EMBEDDING = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${GEMINI_API_KEY}`;

// --- In-memory Vector Store (for demonstration purposes) ---
let documentChunks = []; // Stores objects like { text: "...", embedding: [...] }

// --- Utility Functions (chunkText, getEmbedding, cosineSimilarity, convertWordsToDigits) ---
// These functions remain the same as in the previous code block.
// I'm omitting them here for brevity, but they should be present in your full file.

function chunkText(text, maxTokens = 200, overlapTokens = 50) {
    const words = text.split(/\s+/);
    const chunks = [];
    let currentChunkWords = [];

    for (let i = 0; i < words.length; i++) {
        currentChunkWords.push(words[i]);
        if (currentChunkWords.length >= maxTokens) {
            chunks.push(currentChunkWords.join(' '));
            currentChunkWords = currentChunkWords.slice(maxTokens - overlapTokens);
        }
    }
    if (currentChunkWords.length > 0) {
        chunks.push(currentChunkWords.join(' '));
    }
    return chunks;
}

async function getEmbedding(text) {
    try {
        const response = await axios.post(
            GEMINI_API_URL_EMBEDDING,
            {
                model: "text-embedding-004",
                content: { parts: [{ text: text }] }
            },
            {
                headers: { 'Content-Type': 'application/json' },
                timeout: 10000
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


// --- API Endpoints ---

app.post('/hackrx/upload-pdf', async (req, res) => {
    try {
        const { documentUrl } = req.body;
        if (!documentUrl) {
            return res.status(400).json({ error: 'documentUrl is required in the request body.' });
        }

        console.log(`Processing PDF from: ${documentUrl}`);
        const pdfResponse = await axios.get(documentUrl, { responseType: 'arraybuffer', timeout: 60000 });
        const pdfData = await pdfParse(pdfResponse.data);
        const extractedText = pdfData.text;

        console.log('Extracted Text Length:', extractedText.length);

        const chunks = chunkText(extractedText, 500, 100);
        console.log(`Generated ${chunks.length} chunks.`);

        documentChunks = [];
        const embeddingPromises = chunks.map(async (chunk, index) => {
            try {
                const embedding = await getEmbedding(chunk);
                documentChunks.push({ text: chunk, embedding: embedding });
                console.log(`Embedded chunk ${index + 1}/${chunks.length}`);
            } catch (error) {
                console.error(`Failed to embed chunk ${index + 1}:`, error.message);
            }
        });
        await Promise.all(embeddingPromises);

        console.log(`Finished indexing document with ${documentChunks.length} chunks.`);
        res.status(200).json({
            message: `Document processed and indexed with ${documentChunks.length} chunks.`,
            documentUrl: documentUrl
        });

    } catch (err) {
        console.error('Error processing PDF for indexing:', err.message);
        res.status(500).json({ error: 'Failed to process PDF for indexing', detail: err.message });
    }
});

app.post('/hackrx/run', async (req, res) => {
    const startTime = process.hrtime.bigint();

    try {
        const { questions } = req.body;

        if (!Array.isArray(questions) || questions.length === 0) {
            return res.status(400).json({ error: 'Invalid input format. "questions" (array of strings) are required.' });
        }
        if (!GEMINI_API_KEY) {
            return res.status(500).json({ error: 'API key for Gemini is not configured. Please set GEMINI_API_KEY in your .env file.' });
        }
        if (documentChunks.length === 0) {
            return res.status(400).json({ error: 'No document has been processed yet. Please use the /hackrx/upload-pdf endpoint first to load a document.' });
        }

        const answerPromises = questions.map(async (question) => {
            // 1. Get embedding for the current question
            const questionEmbedding = await getEmbedding(question);

            // 2. Retrieve top N relevant chunks from our indexed document
            const similarities = documentChunks.map(chunk => ({
                chunk: chunk.text,
                similarity: cosineSimilarity(questionEmbedding, chunk.embedding)
            }));

            similarities.sort((a, b) => b.similarity - a.similarity);
            const topN = 5;
            const relevantContextChunks = similarities.slice(0, topN).map(s => s.chunk);
            const context = relevantContextChunks.join('\n\n');
            const MAX_LLM_CONTEXT_LENGTH = 30000;
            const finalContext = context.length > MAX_LLM_CONTEXT_LENGTH ? context.substring(0, MAX_LLM_CONTEXT_LENGTH) : context;

            // --- THIS IS WHERE YOU ADD THE PROMPT CONSTANT ---
            const prompt = {
  contents: [
    {
      role: 'user',
      parts: [
        {
          text: `
Context:
${finalContext}

Instructions:
- Answer strictly based on context above.
- Quote/match the relevant sentence(s) from context to support your answer.
- If answer not found, state "Not found in document."

Question: ${question}
`
        }
      ]
    }
  ]
};

            // --- END OF PROMPT LOCATION ---

            // 4. Make the API call to Google Gemini for this specific question
            try {
                const geminiResponse = await axios.post(
                    GEMINI_API_URL_CONTENT,
                    prompt, // <--- This 'prompt' variable is used here
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
                    .trim();

                cleanedAnswer = convertWordsToDigits(cleanedAnswer);

                return cleanedAnswer;
            } catch (geminiError) {
                console.error(`Error asking Gemini for question "${question}":`, geminiError.response ? geminiError.response.data : geminiError.message);
                return `Error retrieving answer for "${question}".`;
            }
        });

        const answers = await Promise.all(answerPromises);

        const endTime = process.hrtime.bigint();
        const responseTimeMs = Number(endTime - startTime) / 1_000_000;

        res.json({ answers, responseTimeMs: `${responseTimeMs.toFixed(2)} ms` });

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
