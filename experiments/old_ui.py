import streamlit as st
import tempfile
from pathlib import Path

from src.preprocessing.ingestion_preprocessing import HeaderInfoBuilder
from src.preprocessing.markdown_parser import build_tree
from src.embeddings.embedder import chunk_encoder
from src.vector_store.vector_store import FaissVectorStore
from src.generation.llm import LLMResponseGenerator
from src.chunking.semantic_chunking import semantic_chunker
from src.chunking.chunker import text_splitter

import pymupdf
import pymupdf4llm


st.set_page_config(
    page_title="RAG Knowledge System",
    page_icon="📚",
    layout="wide",
)


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "stats" not in st.session_state:
    st.session_state.stats = {}

# The built pipeline objects must live in session_state, not as local
# variables inside the button handler -- Streamlit reruns the whole
# script on every interaction, so locals from a previous run are gone.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm" not in st.session_state:
    st.session_state.llm = None


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def build_knowledge_base(pdf_path: Path):
    """
    Runs the full ingestion pipeline on the given PDF path and returns
    (vector_store, llm, num_semantic_chunks, num_vector_chunks).
    """

    # 1. Dynamic heading detection (font stats -> hdr_info)
    header_builder = HeaderInfoBuilder()
    hdr_info = header_builder.build(str(pdf_path))

    # 2. PDF -> markdown, using the dynamically derived heading map.
    #    Uses the UPLOADED file's path, not a hardcoded one.
    pymupdf4llm.use_layout(False)
    doc = pymupdf.open(str(pdf_path))
    markdown = pymupdf4llm.to_markdown(
        doc,
        show_progress=True,
        hdr_info=hdr_info,
    )

    # 3. Markdown -> hierarchical tree
    markdown_tree = build_tree(text=markdown)

    # 4. Tree -> semantic (leaf/rollup) chunks
    chunks = semantic_chunker([markdown_tree])

    # 5. Split any oversized chunk content further (Level 2).
    #    Appending now happens INSIDE the loop -- every chunk gets
    #    kept, not just the last one.
    final_chunks = []
    for chunk in chunks:
        splitted_chunks = text_splitter(chunk["content"])
        final_chunks.append({
            "title": chunk["title"],
            "chunks": splitted_chunks,
            "breadcrumb": chunk["breadcrumb"],
        })

    # 6. Flatten into (text, metadata) pairs for embedding
    embedding_texts = []
    embedding_metadata = []

    for chunk in final_chunks:
        for text in chunk["chunks"]:
            embedding_texts.append(text)
            embedding_metadata.append({
                "title": chunk["title"],
                "breadcrumb": chunk["breadcrumb"],
                "content": text,
            })

    if not embedding_texts:
        raise ValueError(
            "No text chunks were produced from this PDF. "
            "Check the extraction/chunking pipeline."
        )

    # 7. Embed and store
    embeddings = chunk_encoder(embedding_texts)
    dimension = embeddings.shape[1]

    vector_store = FaissVectorStore(dimension=dimension)
    vector_store.add(embeddings=embeddings, metadata_list=embedding_metadata)

    # 8. LLM for answer generation
    llm = LLMResponseGenerator()

    return vector_store, llm, len(final_chunks), vector_store.total_vectors()


def run_query(question: str, vector_store, llm, k: int = 5):
    """
    Embeds the question, retrieves top-k chunks, and generates an
    answer grounded in that context. Returns (answer_text, sources).
    """
    query_embedding = chunk_encoder([question])
    results = vector_store.search(query_embedding, k=k)[0]

    context = llm.build_context(results)
    prompt = llm.build_prompt(user_question=question, context=context)
    response = llm.generate_response(prompt=prompt)

    # Normalize "distance" -> "score" so the UI has a consistent field
    # to display, regardless of what the store returns it as.
    sources = []
    for r in results:
        sources.append({
            "title": r.get("title", "Untitled"),
            "breadcrumb": r.get("breadcrumb", ""),
            "content": r.get("content", ""),
            "score": r.get("distance"),
        })

        print(r)

    return response.text, sources


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.header("📚 RAG Knowledge System")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF to build the knowledge base.",
    )

    if uploaded_file is not None:
        st.caption(f"Selected: **{uploaded_file.name}**")

        if st.button(
            "Build Knowledge Base",
            type="primary",
            use_container_width=True,
        ):
            tmp_path = None
            try:
                with st.spinner("Building knowledge base... this may take a minute."):

                    # Save the uploaded PDF to a temp file and use THAT
                    # path for every step below.
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = Path(tmp.name)

                    vector_store, llm, num_semantic, num_vectors = (
                        build_knowledge_base(tmp_path)
                    )

                    # Persist across reruns.
                    st.session_state.vector_store = vector_store
                    st.session_state.llm = llm
                    st.session_state.rag_ready = True
                    st.session_state.messages = []
                    st.session_state.stats = {
                        "document": uploaded_file.name,
                        "semantic_chunks": num_semantic,
                        "vector_chunks": num_vectors,
                    }

                st.success("Knowledge base ready. Ask a question below.")

            except Exception as exc:
                st.error(f"Failed to build knowledge base: {exc}")

            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()

    st.divider()

    if st.session_state.rag_ready:
        st.success("🟢 RAG ready")

        stats = st.session_state.stats
        col1, col2 = st.columns(2)
        col1.metric("Semantic chunks", stats.get("semantic_chunks", 0))
        col2.metric("Vector chunks", stats.get("vector_chunks", 0))

        st.caption(f"Document: {stats.get('document', 'Unknown')}")
    else:
        st.info("Upload a PDF and build the knowledge base.")


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------

st.title("📖 RAG Knowledge System")
st.caption(
    "Upload a PDF, build the knowledge base, and ask questions about it."
)


def render_sources(sources):
    with st.expander("Retrieved sources"):
        for source in sources:
            st.markdown(f"**{source.get('title', 'Untitled')}**")

            # breadcrumb may be a plain string ("A > B > C") -- display
            # it directly rather than joining it character-by-character.
            breadcrumb = source.get("breadcrumb")
            if breadcrumb:
                st.caption(str(breadcrumb))

            st.markdown(source.get("content", ""))

            if source.get("score") is not None:
                st.caption(f"Distance: {source['score']:.4f}")

            st.divider()


# -------------------------------------------------------------------
# Chat history
# -------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])


# -------------------------------------------------------------------
# Question
# -------------------------------------------------------------------

question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.rag_ready:
        answer = "Please upload a PDF and build the knowledge base first."
        sources = []
        with st.chat_message("assistant"):
            st.warning(answer)

    else:
        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the knowledge base..."):
                    answer, sources = run_query(
                        question,
                        st.session_state.vector_store,
                        st.session_state.llm,
                        k=5,
                    )

                st.markdown(answer)

                if sources:
                    render_sources(sources)

        except Exception as exc:
            answer = f"RAG error: {exc}"
            sources = []
            st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })