const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const ort = require('onnxruntime-node');
const { AutoTokenizer } = require('@xenova/transformers');

const app = express();
const PORT = 3000;

app.use(express.json());

const GEMINI_API_KEY = 'AIzaSyC3loxj3YwITkUl6gDK3QY53QYvHElLhrw';
const GEMINI_API_URL_CONTENT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;

let localEmbeddingSession = null;
let localTokenizer = null;
const LOCAL_MODEL_PATH = './models/model.onnx';
const LOCAL_TOKENIZER_PATH = './models/tokenizer.json';

async function initializeLocalEmbeddingModel() {
    try {
        localEmbeddingSession = await ort.InferenceSession.create(LOCAL_MODEL_PATH);
        localTokenizer = await AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');
        console.log('Transformer tokenizer loaded.');
    } catch (err) {
        console.error('Tokenizer init error:', err);
    }
}

function chunkText(text, maxTokens = 500, overlapTokens = 100) {
    const chunks = [];
    const paragraphs = text.split(/\n\s*\n/);
    for (const para of paragraphs) {
        if (!para.trim()) continue;
        const words = para.split(/\s+/);
        let chunk = [];
        for (let i = 0; i < words.length; i++) {
            chunk.push(words[i]);
            if (chunk.length >= maxTokens) {
                chunks.push(chunk.join(' '));
                chunk = chunk.slice(-overlapTokens);
            }
        }
        if (chunk.length) chunks.push(chunk.join(' '));
    }
    return chunks;
}

function cosineSimilarity(a, b) {
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    return (Math.sqrt(magA) * Math.sqrt(magB)) ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;
}

function convertWordsToDigits(text) {
    const map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
        'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100'
    };
    return Object.entries(map).reduce((t, [k, v]) => t.replace(new RegExp(`\\b${k}\\b`, 'gi'), v), text);
}

async function getEmbedding(text) {
    if (!localEmbeddingSession || !localTokenizer) {
        throw new Error('Model or tokenizer not initialized');
    }

    const encoding = await localTokenizer.encode(text);
    const input_ids = new ort.Tensor('int64', BigInt64Array.from(encoding.ids.map(BigInt)), [1, encoding.ids.length]);
    const attention_mask = new ort.Tensor('int64', BigInt64Array.from(encoding.attention_mask.map(BigInt)), [1, encoding.attention_mask.length]);
    const token_type_ids = new ort.Tensor('int64', BigInt64Array.from(encoding.type_ids.map(BigInt)), [1, encoding.type_ids.length]);

    const feeds = { input_ids, attention_mask, token_type_ids };
    const output = await localEmbeddingSession.run(feeds);
    const last = output.last_hidden_state.data;
    const mask = encoding.attention_mask;
    const dim = output.last_hidden_state.dims[2];
    const pooled = Array(dim).fill(0);
    let count = 0;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i] === 1) {
            for (let j = 0; j < dim; j++) {
                pooled[j] += last[i * dim + j];
            }
            count++;
        }
    }
    if (count > 0) pooled.forEach((v, i) => pooled[i] = v / count);
    return pooled;
}

app.post('/hackrx/run', async (req, res) => {
    try {
        const { documents, questions } = req.body;
        if (!documents || !Array.isArray(questions)) return res.status(400).json({ error: 'Invalid input.' });

        const pdfBuffer = (await axios.get(documents, { responseType: 'arraybuffer' })).data;
        const text = (await pdfParse(pdfBuffer)).text;
        const chunks = chunkText(text, 500, 100);

        const indexedChunks = await Promise.all(chunks.map(async (chunk) => {
            try {
                const embedding = await getEmbedding(chunk);
                return { text: chunk, embedding };
            } catch { return null; }
        }));

        const validChunks = indexedChunks.filter(Boolean);
        const answers = await Promise.all(questions.map(async (q) => {
            const qEmbed = await getEmbedding(q);
            const ranked = validChunks.map(c => ({
                chunk: c.text,
                sim: cosineSimilarity(qEmbed, c.embedding)
            })).sort((a, b) => b.sim - a.sim);

            const context = ranked.slice(0, 7).map(c => c.chunk).join('\n\n').slice(0, 30000);
            const prompt = {
                contents: [{
                    role: 'user',
                    parts: [{ text: `Context:\n${context}\n\nAnswer: ${q}` }]
                }]
            };

            try {
                const gRes = await axios.post(GEMINI_API_URL_CONTENT, prompt, { headers: { 'Content-Type': 'application/json' } });
                let output = gRes.data.candidates?.[0]?.content?.parts?.[0]?.text || '';
                return convertWordsToDigits(output.trim()
                    .replace(/^\d+\.\s*/, '')
                    .replace(/Answer: /i, '')
                    .replace(/The answer is: /i, '')
                    .replace(/^[-•] /, '')
                    .trim());
            } catch {
                return 'Not found in document.';
            }
        }));

        res.json({ answers });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal server error.', detail: err.message });
    }
});

app.listen(PORT, async () => {
    await initializeLocalEmbeddingModel();
    console.log(`Server running at http://localhost:${PORT}`);
});
