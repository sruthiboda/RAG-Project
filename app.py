import html
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import fitz
import numpy as np
import pdfplumber
import pytesseract
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from pytesseract import TesseractNotFoundError
from sentence_transformers import SentenceTransformer
from transformers import pipeline

try:
    from htmlTemplates import css, bot_template, user_template
except ImportError:
    css, bot_template, user_template = "", None, None


APP_TITLE = "MultiDoc-KBSE"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
QA_MODEL = "deepset/minilm-uncased-squad2"
GEMINI_MODEL = "gemini-1.5-flash"
MIN_PAGE_TEXT_CHARS = 80
MIN_DOC_TEXT_CHARS = 250
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 5
RETRIEVAL_FETCH_K = 20
MIN_RETRIEVAL_SCORE = -1.0
MIN_QA_SCORE = 0.12
NOT_FOUND = "Not found in the uploaded documents."


@dataclass
class PageText:
    source: str
    page: int
    text: str
    method: str


# -------- PDF EXTRACTION --------
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_with_pymupdf(pdf_bytes: bytes, file_name: str) -> list[PageText]:
    pages = []
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for index, page in enumerate(document, start=1):
        text = normalize_text(page.get_text("text") or "")
        pages.append(PageText(file_name, index, text, "text"))
    return pages


def extract_with_pdfplumber(pdf_bytes: bytes, file_name: str) -> list[PageText]:
    pages = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = normalize_text(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
            pages.append(PageText(file_name, index, text, "pdfplumber"))
    return pages


def ocr_page(page, file_name: str, page_number: int) -> PageText:
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png")))
    text = normalize_text(pytesseract.image_to_string(image, config="--psm 6"))
    return PageText(file_name, page_number, text, "ocr")


def extract_pdf_pages(pdf_file) -> tuple[list[PageText], list[str]]:
    file_name = pdf_file.name
    pdf_bytes = pdf_file.getvalue()
    warnings = []

    try:
        pages = extract_with_pymupdf(pdf_bytes, file_name)
    except Exception as exc:
        warnings.append(f"{file_name}: PyMuPDF extraction failed ({exc}). Trying pdfplumber.")
        try:
            pages = extract_with_pdfplumber(pdf_bytes, file_name)
        except Exception as plumber_exc:
            return [], [f"{file_name}: could not read PDF ({plumber_exc})."]

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for index, page_text in enumerate(pages):
            if len(page_text.text) >= MIN_PAGE_TEXT_CHARS:
                continue
            try:
                ocr_text = ocr_page(document[index], file_name, index + 1)
                if len(ocr_text.text) > len(page_text.text):
                    pages[index] = ocr_text
            except TesseractNotFoundError:
                warnings.append(
                    f"{file_name}: OCR is not available. Add tesseract-ocr in packages.txt on Streamlit Cloud."
                )
                break
            except Exception as exc:
                warnings.append(f"{file_name} page {index + 1}: OCR failed ({exc}).")
    except Exception as exc:
        warnings.append(f"{file_name}: OCR setup failed ({exc}).")

    pages = [page for page in pages if page.text.strip()]
    return pages, warnings


def load_documents(pdf_docs):
    all_pages = []
    all_warnings = []

    for pdf in pdf_docs:
        pages, warnings = extract_pdf_pages(pdf)
        all_pages.extend(pages)
        all_warnings.extend(warnings)

    if all_warnings:
        st.warning("\n".join(all_warnings))

    readable_by_file = {}
    for page in all_pages:
        readable_by_file.setdefault(page.source, 0)
        readable_by_file[page.source] += len(page.text)

    weak_files = [f"{name} ({count} characters)" for name, count in readable_by_file.items() if count < MIN_DOC_TEXT_CHARS]
    if weak_files:
        st.warning(
            "Some PDFs still have very little readable text after extraction/OCR. Answers may be limited:\n"
            + "\n".join(weak_files)
        )

    if all_pages:
        st.info(f"Processed {len(all_pages)} pages from {len(readable_by_file)} PDF file(s).")

    return all_pages


# -------- CHUNKING --------
def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = normalize_text(text)
    if len(text) <= size:
        return [text] if text else []

    units = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = f"{current} {unit}".strip()
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:].strip()
        if len(unit) > size:
            for start in range(0, len(unit), max(1, size - overlap)):
                chunks.append(unit[start : start + size].strip())
            current = ""
        else:
            current = f"{current} {unit}".strip()

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def build_chunks(pages: list[PageText]) -> list[dict]:
    chunks = []
    for page in pages:
        for part in split_text(page.text):
            chunks.append(
                {
                    "text": part,
                    "source": page.source,
                    "page": page.page,
                    "method": page.method,
                }
            )
    return chunks


# -------- MODELS --------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)


def get_google_api_key():
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def load_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(os.getenv("GEMINI_MODEL", GEMINI_MODEL))


# -------- KNOWLEDGE BASE --------
def build_knowledge_base(chunks: list[dict]):
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    embeddings = load_embedder().encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {"chunks": chunks, "embeddings": embeddings}


def query_terms(question: str) -> set[str]:
    stop_words = {
        "what", "is", "the", "a", "an", "of", "to", "in", "and", "or", "for", "my", "your",
        "explain", "about", "describe", "give", "tell", "from", "pdf",
    }
    return {term for term in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(term) > 2 and term not in stop_words}


def keyword_score(question: str, text: str) -> float:
    terms = query_terms(question)
    if not terms:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for term in terms if term in lowered)
    normalized_question = question.lower().strip(" ?")
    exact_phrase_bonus = 1.0 if normalized_question in lowered else 0.0
    phrase_bonus = 0.0
    for phrase in ("working principle", "rf energy harvesting", "design of rectenna", "rectenna"):
        if phrase in normalized_question and phrase in lowered[:350]:
            phrase_bonus += 0.8
    definition_bonus = 0.0
    if normalized_question.startswith(("what is", "what are", "define")):
        for term in terms:
            term_pattern = re.escape(term)
            if re.search(rf"\b(?:a|an|the)?\s*{term_pattern}\s+(?:is|are|refers|means)\b", lowered[:500]):
                definition_bonus += 1.0
    heading_bonus = 0.0
    heading_zone = lowered[:180]
    if any(term in heading_zone for term in terms):
        heading_bonus = 0.5
    return (hits / len(terms)) + exact_phrase_bonus + phrase_bonus + definition_bonus + heading_bonus


def mmr_select(candidate_indexes, embeddings, scores, top_k: int, lambda_mult: float = 0.70):
    selected = []
    remaining = list(candidate_indexes)
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda idx: scores[idx])
        else:
            best = max(
                remaining,
                key=lambda idx: lambda_mult * scores[idx]
                - (1 - lambda_mult) * max(float(np.dot(embeddings[idx], embeddings[chosen])) for chosen in selected),
            )
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve_context(question: str, knowledge_base: dict, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    query_embedding = load_embedder().encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    embeddings = knowledge_base["embeddings"]
    semantic_scores = np.dot(embeddings, query_embedding)

    combined_scores = []
    for index, chunk in enumerate(knowledge_base["chunks"]):
        combined = float(semantic_scores[index]) + 0.35 * keyword_score(question, chunk["text"])
        combined_scores.append(combined)

    fetch_k = min(RETRIEVAL_FETCH_K, len(combined_scores))
    candidate_indexes = np.argsort(combined_scores)[::-1][:fetch_k]
    selected_indexes = mmr_select(candidate_indexes, embeddings, combined_scores, top_k)

    results = []
    for index in selected_indexes:
        item = dict(knowledge_base["chunks"][index])
        item["score"] = float(combined_scores[index])
        results.append(item)
    return results


def broaden_context(question: str, knowledge_base: dict, retrieved_contexts: list[dict]) -> list[dict]:
    chunks = knowledge_base["chunks"]
    if len(chunks) <= 12:
        return chunks

    q = question.lower()
    if any(word in q for word in ["summary", "summarize", "summarise", "overview", "about", "skills", "projects", "experience"]):
        selected = []
        seen = set()
        for context in retrieved_contexts + chunks[:4]:
            key = (context["source"], context["page"], context["text"][:80])
            if key not in seen:
                selected.append(context)
                seen.add(key)
        return selected[:12]

    return retrieved_contexts


# -------- DIRECT FIELD ANSWERS --------
def clean_value(value):
    value = re.sub(r"\s+", " ", str(value)).strip(" :-")
    return value if value and value != "-" else ""


def first_match(patterns: Iterable[str], text: str):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            groups = [clean_value(group) for group in match.groups()]
            groups = [group for group in groups if group]
            if groups:
                return " ".join(groups)
    return None


def uploaded_file_names(full_text: str) -> list[str]:
    names = re.findall(r"Source file:\s*([^\n]+)", full_text)
    unique_names = []
    for name in names:
        name = name.strip()
        if name and name not in unique_names:
            unique_names.append(name)
    return unique_names


def helpful_not_found(question: str, full_text: str) -> str:
    files = uploaded_file_names(full_text)
    checked = f" I checked: {', '.join(files)}." if files else ""
    q = question.lower()
    if any(word in q for word in ["father", "mother", "registration", "roll", "application number", "exam centre", "exam center"]):
        return (
            f"{NOT_FOUND}{checked} This looks like a personal/application field, so upload the actual "
            "application form, admit card, or scorecard that contains that field. A job notification or resume usually will not contain it."
        )
    return f"{NOT_FOUND}{checked}"


def answer_common_field(question: str, full_text: str):
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
        "name": ["my name", "full name", "candidate name", "applicant name"],
        "registration": ["registration", "application number"],
        "roll": ["roll"],
        "exam centre": ["exam centre", "exam center", "examination centre", "examination center", "centre", "center"],
        "date of birth": ["date of birth", "dob", "birth"],
        "email": ["email", "mail id"],
        "mobile": ["mobile", "phone", "contact number"],
    }

    for field, keywords in aliases.items():
        if any(keyword in normalized_question for keyword in keywords):
            return first_match(field_patterns[field], compact_text) or helpful_not_found(question, full_text)
    return None


# -------- ANSWERING --------
def format_context(contexts: list[dict]) -> str:
    blocks = []
    for idx, context in enumerate(contexts, start=1):
        blocks.append(
            f"[Source {idx}: {context['source']}, page {context['page']}, extraction={context['method']}]\n{context['text']}"
        )
    return "\n\n".join(blocks)


def cite_sources(contexts: list[dict]) -> str:
    seen = []
    for context in contexts[:4]:
        label = f"{context['source']} p.{context['page']}"
        if label not in seen:
            seen.append(label)
    return "; ".join(seen)


def answer_with_gemini(question: str, contexts: list[dict]):
    api_key = get_google_api_key()
    if not api_key:
        return None

    model = load_gemini_model(api_key)
    prompt = f"""You are a careful PDF question-answering assistant.
Answer ONLY from the retrieved PDF context below.
Do not use outside knowledge and do not guess.
If the retrieved context does not contain the answer, reply exactly: {NOT_FOUND}
If the user asks for a summary or what the PDF is about, summarize only the main content visible in the retrieved context.
If the user asks for a definition, use the chunk that directly defines the term.
If the user asks for a working principle/process, use the chunk that describes the principle or steps, not experimental setup unless the setup is explicitly asked.
If the user asks for skills, projects, education, or experience, extract the relevant bullet points or phrases from the context.
Explain in detail when the question asks to explain. Include source page citations in parentheses.

PDF context:
{format_context(contexts)}

Question: {question}
"""
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0, "max_output_tokens": 700},
    )
    return (response.text or "").strip()


def answer_with_extractive_qa(question: str, contexts: list[dict]):
    qa_model = load_qa_model()
    best_answer = None
    best_score = 0.0
    best_context = None

    for context in contexts:
        result = qa_model(question=question, context=context["text"])
        answer = clean_value(result.get("answer", ""))
        score = float(result.get("score", 0.0))
        if answer and score > best_score:
            best_answer = answer
            best_score = score
            best_context = context

    if not best_answer or best_score < MIN_QA_SCORE:
        return NOT_FOUND

    return f"{best_answer} ({best_context['source']} p.{best_context['page']})"


def summarize_documents(contexts: list[dict], full_text: str):
    summary_contexts = contexts
    if not summary_contexts:
        summary_contexts = [
            {"source": "uploaded PDFs", "page": "mixed", "method": "text", "text": full_text[:7000]}
        ]

    api_answer = answer_with_gemini("What is this PDF about? Give a clear summary of the uploaded document.", summary_contexts)
    if api_answer:
        return api_answer

    text = full_text[:4500]
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    summary_lines = [clean_value(sentence) for sentence in sentences if len(clean_value(sentence)) > 45][:8]
    if not summary_lines:
        return NOT_FOUND
    return "Summary:\n- " + "\n- ".join(summary_lines)


def is_summary_question(question: str) -> bool:
    q = question.lower().strip()
    summary_phrases = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "what is this pdf about",
        "what the pdf is about",
        "what is the pdf about",
        "what is document about",
        "explain about",
        "explain the pdf",
        "describe the pdf",
    ]
    return any(phrase in q for phrase in summary_phrases)


def answer_question(question: str, knowledge_base: dict, full_text: str):
    direct_answer = answer_common_field(question, full_text)
    if direct_answer:
        return direct_answer

    contexts = retrieve_context(question, knowledge_base)
    expanded_contexts = broaden_context(question, knowledge_base, contexts)

    if is_summary_question(question):
        return summarize_documents(expanded_contexts, full_text)

    try:
        api_answer = answer_with_gemini(question, expanded_contexts)
        if api_answer:
            return api_answer
    except Exception as exc:
        st.warning(f"Gemini answer failed, using local fallback: {exc}")

    if not contexts:
        return helpful_not_found(question, full_text)

    answer = answer_with_extractive_qa(question, contexts)
    if answer == NOT_FOUND:
        base_message = helpful_not_found(question, full_text)
        return f"{base_message} Closest source pages checked: {cite_sources(contexts)}"
    return answer


# -------- CHAT RENDERING --------
def render_message(template, role, message):
    safe_message = html.escape(message).replace("\n", "<br>")
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
        answer = answer_question(user_question, st.session_state.knowledge_base, st.session_state.raw_text)

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
        .stButton > button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h1 class='main-title'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='main-subtitle'>Upload PDFs, including scanned PDFs, and ask grounded questions with page sources.</p>",
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

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if pdf_docs:
            for file in pdf_docs:
                st.write(f"- {file.name}")

        if st.button("Process", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Extracting text and building search index..."):
                try:
                    pages = load_documents(pdf_docs)
                    chunks = build_chunks(pages)
                    raw_text = "\n\n".join(
                        f"Source file: {page.source}\nPage {page.page} ({page.method})\n{page.text}" for page in pages
                    )
                    if len(raw_text) < MIN_DOC_TEXT_CHARS:
                        st.error("Not enough readable text was found. Try a clearer PDF or enable OCR support.")
                        return

                    st.session_state.knowledge_base = build_knowledge_base(chunks)
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

        if get_google_api_key():
            st.caption("Answer mode: Gemini + retrieval")
        else:
            st.caption("Answer mode: free local fallback")

    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    show_chat_history()


if __name__ == "__main__":
    main()
