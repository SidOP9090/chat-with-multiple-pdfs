# Chat with Multiple PDFs — RAG-based Document Q&A

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload multiple PDF
documents and ask natural language questions about their content. Built using free,
open-source embeddings and Google's Gemini API for generation.

## How it works

The system follows the standard RAG pipeline:

1. **Ingestion** — Multiple PDFs are uploaded and text is extracted page by page.
2. **Chunking** — Extracted text is split into overlapping chunks (1000 characters,
   200 overlap) to preserve context across boundaries.
3. **Embedding** — Each chunk is converted into a vector using HuggingFace's
   `all-MiniLM-L6-v2` sentence-transformer model, then stored in a FAISS vector
   index for fast similarity search.
4. **Retrieval** — When a user asks a question, the most semantically similar chunks
   are retrieved from the FAISS index.
5. **Generation** — Retrieved context is passed to Google's Gemini 2.5 Flash model
   along with the question and conversation history, producing a grounded answer.

The app maintains conversation memory, so follow-up questions retain context from
earlier in the session.

## Tech stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain (ConversationalRetrievalChain)
- **Embeddings:** HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) — free, runs locally
- **Vector store:** FAISS (Facebook AI Similarity Search)
- **LLM:** Google Gemini 2.5 Flash (free tier)
- **PDF parsing:** PyPDF2

## Setup

```bash
git clone https://github.com/<your-username>/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs
pip install -r requirements.txt
```

Create a `.env` file (use `.env.example` as a template) and add your free Gemini API key:

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Run

```bash
streamlit run app.py
```

Upload PDFs in the sidebar, click **Process**, then ask questions in the chat input.

## Author

Siddhartha Gupta — B.Tech Mechanical Engineering, IIT Ropar