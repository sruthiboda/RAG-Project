# MultiDoc-KBSE

MultiDoc-KBSE is a Streamlit app that lets users upload one or more PDF files and ask questions about their contents. It uses a Retrieval-Augmented Generation pipeline with PDF text extraction, text chunking, Hugging Face embeddings, FAISS vector search, and a Flan-T5 language model.

## Live Demo

Use the deployed app here:

https://multi-docs-kbse-huzqndko7noyg6zsu6ehpk.streamlit.app/

## Features

- Chat with multiple PDF documents
- Extract text from uploaded PDFs
- Split document text into searchable chunks
- Create semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store and search vectors with FAISS
- Generate answers with `google/flan-t5-base`
- Keep conversational context during a session
- Handles unreadable/scanned PDFs with clear error messages

## How It Works

1. Upload one or more PDF files.
2. Click **Process**.
3. The app extracts readable text from each PDF.
4. Text is split into overlapping chunks.
5. Chunks are embedded and stored in a FAISS vector index.
6. A user question is matched with the most relevant chunks.
7. Flan-T5 generates an answer using the retrieved context.

## Project Structure

```text
RAG-Project/
|-- app.py
|-- htmlTemplates.py
|-- requirements.txt
|-- runtime.txt
|-- readme.md
|-- docs/
|   |-- Knowledgebase_img.png
|   |-- PDF-LangChain.jpg
|-- Screenshot 2026-04-02 232711.png
```

## Requirements

Use Python 3.10 for best compatibility with Streamlit Cloud and the pinned ML dependencies.

## Local Setup

```bash
git clone https://github.com/sruthiboda/RAG-Project.git
cd RAG-Project
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Upload one or more text-based PDF files.
3. Click **Process**.
4. Ask questions such as:
   - Summarize the document.
   - What are the key findings?
   - Explain this in simple terms.

## Notes

- Scanned image-only PDFs usually do not contain extractable text. Run OCR first if the app says no readable text was found.
- The first run can be slow because Hugging Face models need to download and load.
- The app uses open-source Hugging Face models, so no OpenAI API key is required.
