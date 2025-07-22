const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const app = express();
const PORT = 3000;

app.use(express.json());

const GEMINI_API_KEY = 'AIzaSyAGMeiEKbp2ELpbF3mmaJ6660qzwqBHnqM'

const GEMINI_API_URL_CONTENT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;
const GEMINI_API_URL_EMBEDDING = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${GEMINI_API_KEY}`;
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

app.post('/hackrx/run', async (req, res) => {
    
    try {
        const { documents, questions } = req.body;

        if (questions.length === 0) {
            return res.json({ answers: [] });
        }

        console.log(`Processing PDF from: ${documents}`);
        const pdfResponse = await axios.get(documents, { responseType: 'arraybuffer', timeout: 90000 });
        const pdfData = await pdfParse(pdfResponse.data);
        const extractedText = pdfData.text;
        console.log('Extracted Text Length:', extractedText.length);

        const chunks = chunkText(extractedText, 500, 100);
        console.log(`Generated ${chunks.length} chunks.`);

        const documentChunks = []; // Local to this request
        await Promise.all(chunks.map(async (chunk, index) => {
            try {
                const embedding = await getEmbedding(chunk);
                documentChunks.push({ text: chunk, embedding: embedding });
            } catch {
                console.log(`Failed to embed chunk ${index + 1}:`);
            }
        }));
        console.log(`Finished embedding ${documentChunks.length} chunks for this request.`);
        const answerPromises = questions.map(async (question) => {
            const questionEmbedding = await getEmbedding(question);
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
            const prompt = {
                contents: [
                    {
                        role: 'user',
                        parts: [
                            {
                                text: `Context:\n${finalContext}\n\nAnswer the following question based ONLY on the provided document context. Your answer MUST be:
1. Extremely concise and direct.
2. Contain ONLY the requested information.
3. Use digits for all numbers (e.g., "36" not "thirty-six").
4. Do NOT include any introductory phrases, explanations, or concluding remarks.
5. If the answer is a definition, provide the exact definition as found in the text.
6. If the answer is not explicitly found in the document context, state "Not found in document."
7. Give all of the answers right and also under 10-20 words max.

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

                // Post-process the output for cleaning and number conversion
                let cleanedAnswer = output.trim();
                cleanedAnswer = cleanedAnswer
                    .replace(/^\d+\.\s*/, '')      // Remove leading numbers like "1. "
                    .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown characters
                    .replace(/\(\d+\.\d+\)/g, '')     // Remove parenthesized numbers (e.g., page numbers)
                    .replace(/^- /, '')             
                    .replace(/The answer is: /i, '')
                    .replace(/Based on the document, /i, '')
                    .replace(/According to the policy, /i, '')
                    .replace(/It states that /i, '')
                    .replace(/Here is the answer: /i, '')
                    .replace(/The document states: /i, '')
                    .replace(/Answer: /i, '') // Added this
                    .replace(/The policy states: /i, '') // Added this
                    .trim();
                cleanedAnswer = convertWordsToDigits(cleanedAnswer);

                return cleanedAnswer;
            } catch {
                console.log('Gemini request failed:');
                return 'Not found in document.';
            }
        });

        const answers = await Promise.all(answerPromises);
        res.json({ answers });

    } catch {
        return res.json({ answers: [] });
    }
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});
