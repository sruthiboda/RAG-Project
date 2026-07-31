# MultiDoc-KBSE

MultiDoc-KBSE is a Streamlit PDF question-answering app. It extracts text from uploaded PDFs, uses OCR for scanned/image pages, retrieves the most relevant page chunks, and answers with source page references.

## Best Accuracy Mode

For the most accurate answers, add an OpenAI key in Streamlit Cloud secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

If no key is provided, the app still works with a free local extractive QA fallback, but complex reasoning and messy PDFs will be less reliable.

## Features

- Upload and chat with multiple PDFs
- Extracts text with PyMuPDF
- Falls back to pdfplumber when needed
- Uses OCR through Tesseract for scanned/image PDFs
- Builds semantic search with Sentence Transformers
- Answers only from retrieved PDF context
- Shows source file and page number for answers
- Returns `Not found in the uploaded documents` instead of guessing
- Supports common form fields such as name, father name, registration number, roll number, exam centre, DOB, email, and mobile number

## Tech Stack

- Streamlit
- PyMuPDF
- pdfplumber
- pytesseract + Tesseract OCR
- Sentence Transformers
- Transformers extractive QA fallback
- Optional OpenAI answer generation

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

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For OCR locally, install Tesseract OCR separately:

- Windows: install Tesseract from UB Mannheim builds and add it to PATH
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
- macOS: `brew install tesseract`

Run the app:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

The repo includes `packages.txt`, so Streamlit Cloud installs Tesseract automatically.

After pushing changes:

1. Open Streamlit Cloud.
2. Go to **Manage app**.
3. Click **Clear cache**.
4. Click **Reboot app**.
5. Upload PDFs and click **Process**.

## Notes

No PDF assistant can guarantee perfect answers for every possible PDF. Accuracy depends on PDF quality, OCR quality, and whether the answer is actually present in the document. This app is designed to avoid hallucination by refusing to answer when the supporting text is not found.
