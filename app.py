import html
import re
from typing import Iterable

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer
from transformers import pipeline

try:
    from htmlTemplates import css, bot_template, user_template
except ImportError:
    css, bot_template, user_template = "", None, None


APP_TITLE = "MultiDoc-KBSE"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/minilm-uncased-squad2"
MIN_READABLE_CHARS_PER_FILE = 300
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
MIN_QA_SCORE = 0.12


# -------- PDF TEXT --------
def get_pdf_text(pdf_docs):
    text_parts = []
    skipped_files = []
    low_text_files = []
    readable_files = []

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            file_pages = []
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    file_pages.append(f"Page {page_number}:\n{page_text.strip()}")

            file_text = "\n\n".join(file_pages).strip()
            if len(file_text) < MIN_READABLE_CHARS_PER_FILE:
                low_text_files.append(f"{pdf.name} ({len(file_text)} readable characters)")
                continue

            readable_files.append(pdf.name)
            text_parts.append(f"Source file: {pdf.name}\n{file_text}")
        except (PdfReadError, ValueError, OSError) as exc:
            skipped_files.append(f"{pdf.name}: {exc}")

    if skipped_files:
        st.warning("Some files could not be read:\n" + "\n".join(skipped_files))

    if low_text_files:
        st.warning(
            "These PDFs were skipped because they do not contain enough selectable text. "
            "They may be scanned/image PDFs, so run OCR first:\n"
            + "\n".join(low_text_files)
        )

    if readable_files:
        st.info("Processed readable text from: " + ", ".join(readable_files))

    return "\n\n".join(text_parts).strip()


# -------- TEXT CHUNKS --------
def get_text_chunks(text):
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{paragraph}".strip()
            continue

        if current:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:]

        if len(paragraph) <= CHUNK_SIZE:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            for start in range(0, len(paragraph), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(paragraph[start : start + CHUNK_SIZE])
            current = ""

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]


# -------- MODELS --------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)


# -------- KNOWLEDGE BASE --------
def build_knowledge_base(text_chunks):
    if not text_chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    embeddings = load_embedder().encode(
        text_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {"chunks": text_chunks, "embeddings": embeddings}


def retrieve_context(question, knowledge_base, top_k=6):
    query_embedding = load_embedder().encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = np.dot(knowledge_base["embeddings"], query_embedding)
    top_indexes = np.argsort(scores)[::-1][:top_k]
    return [knowledge_base["chunks"][index] for index in top_indexes if scores[index] > 0.12]


# -------- DIRECT FIELD ANSWERS --------
def clean_value(value):
    value = re.sub(r"\s+", " ", value).strip(" :-")
    return value if value and value != "-" else "Not found in the uploaded documents."


def first_match(patterns: Iterable[str], text: str):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            groups = [clean_value(group) for group in match.groups() if clean_value(group)]
            groups = [group for group in groups if group != "Not found in the uploaded documents."]
            if groups:
                return " ".join(groups)
    return None


def answer_common_field(question, full_text):
    normalized_question = question.lower()
    compact_text = re.sub(r"\s+", " ", full_text)

    field_patterns = {
        "father": [
            r"Father'?s First Name\s*:\s*([^:\n]+?)\s+Father'?s Middle Name\s*:\s*([^:\n]+?)\s+Father'?s Last Name\s*:\s*([^:\n]+?)(?=\s+(?:Mother|Spouse|Address|Uploaded|Captured|$))",
            r"Father'?s Name\s*:\s*([^:\n]+?)(?=\s+[A-Z][A-Za-z /']+\s*:|$)",
        ],
        "mother": [
            r"Mother'?s First Name\s*:\s*([^:\n]+?)\s+Mother'?s Middle Name\s*:\s*([^:\n]+?)\s+Mother'?s Last Name\s*:\s*([^:\n]+?)(?=\s+(?:Spouse|Address|Uploaded|Captured|$))",
            r"Mother'?s Name\s*:\s*([^:\n]+?)(?=\s+[A-Z][A-Za-z /']+\s*:|$)",
        ],
        "name": [
            r"Full Name\s*:\s*([^:\n]+?)(?=\s+(?:Category|Date of Birth|Gender|Father|$))",
            r"Candidate'?s Name\s*:\s*([^:\n]+?)(?=\s+[A-Z][A-Za-z /']+\s*:|$)",
        ],
        "registration": [r"Registration Number\s*:\s*([A-Z0-9/-]+)"],
        "roll": [r"Roll Number\s*:\s*([A-Z0-9/-]+)", r"Roll No\.?\s*:\s*([A-Z0-9/-]+)"],
        "exam centre": [
            r"Centre of Examination\s*:\s*([^:\n]+?)(?=\s+(?:I intend|Examination|State|ID Proof|$))",
            r"Exam(?:ination)? Centre\s*:\s*([^:\n]+?)(?=\s+[A-Z][A-Za-z /']+\s*:|$)",
        ],
        "date of birth": [r"Date of Birth\s*:\s*([0-9?/-]+)"],
        "email": [r"Email ID\s*:\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})"],
        "mobile": [r"Mobile Number\s*:\s*([+?0-9\s-]+?)(?=\s+(?:Alternative|Email|$))"],
    }

    aliases = {
        "father": ["father", "dad"],
        "mother": ["mother", "mom"],
        "name": ["my name", "full name", "candidate name"],
        "registration": ["registration", "application number"],
        "roll": ["roll"],
        "exam centre": ["exam centre", "exam center", "examination centre", "examination center", "centre", "center"],
        "date of birth": ["date of birth", "dob", "birth"],
        "email": ["email", "mail id"],
        "mobile": ["mobile", "phone", "contact number"],
    }

    for field, keywords in aliases.items():
        if any(keyword in normalized_question for keyword in keywords):
            answer = first_match(field_patterns[field], compact_text)
            if answer:
                if field == "roll" and not answer:
                    return "Not found in the uploaded documents."
                return answer
    return None


# -------- QUESTION ANSWERING --------
def answer_question(question, knowledge_base, full_text):
    direct_answer = answer_common_field(question, full_text)
    if direct_answer:
        return direct_answer

    contexts = retrieve_context(question, knowledge_base)
    if not contexts:
        return "Not found in the uploaded documents."

    qa_model = load_qa_model()
    best_answer = None
    best_score = 0.0

    for context in contexts:
        result = qa_model(question=question, context=context)
        answer = clean_value(result.get("answer", ""))
        score = float(result.get("score", 0.0))
        if answer and answer != "Not found in the uploaded documents." and score > best_score:
            best_answer = answer
            best_score = score

    if not best_answer or best_score < MIN_QA_SCORE:
        return "Not found in the uploaded documents."

    return best_answer


# -------- CHAT RENDERING --------
def render_message(template, role, message):
    safe_message = html.escape(message)
    st.write(template.replace("{{ROLE}}", role).replace("{{MSG}}", safe_message), unsafe_allow_html=True)


def show_chat_history():
    for message in st.session_state.chat_history:
        if user_template and bot_template:
            template = user_template if message["role"] == "You" else bot_template
            render_message(template, message["role"], message["content"])
        else:
            st.markdown(f"**{message['role']}:** {message['content']}")


def handle_userinput(user_question):
    if st.session_state.knowledge_base is None:
        st.warning("Please upload and process documents first.")
        return

    with st.spinner("Searching the uploaded PDFs..."):
        answer = answer_question(
            user_question,
            st.session_state.knowledge_base,
            st.session_state.raw_text,
        )

    st.session_state.chat_history.append({"role": "You", "content": user_question})
    st.session_state.chat_history.append({"role": "Bot", "content": answer})


# -------- MAIN --------
def main():
    load_dotenv()

    st.set_page_config(page_title="KnowledgeBase.com", page_icon="📚", layout="wide")

    st.markdown(
        """
        <style>
        body { background-color: #0e1117; }
        .main-title { text-align: center; margin-bottom: 0.25rem; }
        .main-subtitle { text-align: center; margin-bottom: 2rem; color: #9aa4b2; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h1 class='main-title'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='main-subtitle'>Ask exact questions from readable PDF text.</p>",
        unsafe_allow_html=True,
    )

    if css:
        st.write(css, unsafe_allow_html=True)

    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    show_chat_history()

    with st.sidebar:
        st.subheader("Your Documents")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if pdf_docs:
            for file in pdf_docs:
                st.write(f"- {file.name}")

        if st.button("Process", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing documents..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error(
                            "No readable text was found. Use a text-based PDF, or run OCR on scanned/image PDFs first."
                        )
                        return

                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.knowledge_base = build_knowledge_base(text_chunks)
                    st.session_state.raw_text = raw_text
                    st.session_state.chat_history = []
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    return

            st.success("Processing complete. You can ask questions now.")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.knowledge_base = None
            st.session_state.raw_text = ""
            st.success("Chat cleared.")


if __name__ == "__main__":
    main()
