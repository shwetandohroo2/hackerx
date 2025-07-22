const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const app = express();
const PORT = 3000;

app.use(express.json());

function chunkLongText(text, chunkSize = 100000) {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize) {
    chunks.push(text.slice(i, i + chunkSize));
  }
  return chunks;
}

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyAaK_g64tXlaXIhFzxj-eUT9RzprG7n3xM';

async function summarizeWithGemini(chunk) {
  const prompt = {
    contents: [
      {
        role: 'user',
        parts: [
          {
            text: `Context:\n${chunk}\n\nCompress the following text while keeping all important details. Make it shorter and information-dense but do not miss any information, only shorten all the sentences by removing the unnecessary part of that sentence. Return the result in a paragraph form.`
          }
        ]
      }
    ]
  };

  const response = await axios.post(GEMINI_API_URL, prompt, {
    headers: { 'Content-Type': 'application/json' }
  });

  return response.data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

app.post('/ask', async (req, res) => {
  try {
    const { documents, questions } = req.body;

    if (!documents || !Array.isArray(questions)) {
      return res.status(400).json({ error: 'Invalid input format.' });
    }

    const pdfResponse = await axios.get(documents, { responseType: 'arraybuffer' });
    const pdfData = await pdfParse(pdfResponse.data);
    const extractedText = pdfData.text;

    console.log('Extracted Text Length:', extractedText.length);

    let finalSummary = '';

    if (extractedText.length > 100000) {
      const chunks = chunkLongText(extractedText, 100000);
      console.log(`Summarizing ${chunks.length} chunks in parallel...`);
      const summaries = await Promise.all(chunks.map(summarizeWithGemini));
      finalSummary = summaries.join('\n\n');
    } else {
      finalSummary = await summarizeWithGemini(extractedText);
    }

    console.log('Final Summary Length:', finalSummary.length);

    const prompt = {
      contents: [
        {
          role: 'user',
          parts: [
            {
              text:
                `Context:\n${finalSummary}\n\nYou are a smart RAG-enabled bot. Answer the following questions based on the above document:\n` +
                `Do not provide answers in key-value pairs. Just give the answers in order directly in single lines.\n\n` +
                questions.map((q, i) => `${i + 1}. ${q}`).join('\n')
            }
          ]
        }
      ]
    };

    const geminiResponse = await axios.post(GEMINI_API_URL, prompt, {
      headers: { 'Content-Type': 'application/json' }
    });

    const output = geminiResponse.data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response from Gemini.';

    const answerLines = output.split(/\n(?=\d+\.\s)/);
    const answers = answerLines
      .map(line =>
        line
          .replace(/^\d+\.\s*/, '')
          .replace(/\*\*(.*?)\*\*/g, '$1')
          .replace(/\(\d+\.\d+\)/g, '')
          .trim()
      )
      .filter(Boolean);

    res.json({ answers });

  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Something went wrong', detail: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
