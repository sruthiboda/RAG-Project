import html

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from transformers import pipeline

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS

try:
    from htmlTemplates import css, bot_template, user_template
except ImportError:
    css, bot_template, user_template = "", None, None


APP_TITLE = "MultiDoc-KBSE"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"


# -------- PDF TEXT --------
def get_pdf_text(pdf_docs):
    text_parts = []
    skipped_files = []

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
        except (PdfReadError, ValueError, OSError) as exc:
            skipped_files.append(f"{pdf.name}: {exc}")

    if skipped_files:
        st.warning("Some files could not be read:\n" + "\n".join(skipped_files))

    return "\n".join(text_parts).strip()


# -------- TEXT CHUNKS --------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return [chunk for chunk in text_splitter.split_text(text) if chunk.strip()]


# -------- VECTOR STORE --------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    return FAISS.from_texts(texts=text_chunks, embedding=load_embeddings())


# -------- CONVERSATION CHAIN --------
@st.cache_resource(show_spinner=False)
def load_llm():
    text_generation_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_length=512,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=load_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
    )


# -------- CHAT RENDERING --------
def render_message(template, message):
    safe_message = html.escape(message.content)
    st.write(template.replace("{{MSG}}", safe_message), unsafe_allow_html=True)


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process documents first.")
        return

    try:
        response = st.session_state.conversation.invoke({"question": user_question})
    except Exception as exc:
        st.error(f"Could not generate an answer: {exc}")
        return

    st.session_state.chat_history = response.get("chat_history", [])

    for i, message in enumerate(st.session_state.chat_history):
        if user_template and bot_template:
            render_message(user_template if i % 2 == 0 else bot_template, message)
        elif i % 2 == 0:
            st.markdown(f"**You:** {message.content}")
        else:
            st.markdown(f"**Bot:** {message.content}")


# -------- MAIN --------
def main():
    load_dotenv()

    st.set_page_config(page_title="KnowledgeBase.com", page_icon="KB", layout="wide")

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
        "<p class='main-subtitle'>Ask questions across multiple PDF documents.</p>",
        unsafe_allow_html=True,
    )

    if css:
        st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

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
                            "No readable text was found. Try a text-based PDF, or run OCR on scanned PDFs first."
                        )
                        return

                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = []
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    return

            st.success("Processing complete. You can ask questions now.")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.success("Chat cleared.")


if __name__ == "__main__":
    main()
