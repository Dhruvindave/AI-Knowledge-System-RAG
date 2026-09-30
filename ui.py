import html
import re
import tempfile
import traceback
from pathlib import Path
import sys
import streamlit as st
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ingestion_preprocessing import HeaderInfoBuilder
from src.preprocessing.markdown_parser import build_tree
from src.embeddings.embedder import chunk_encoder
from src.vector_store.vector_store import FaissVectorStore
from src.generation.llm import LLMResponseGenerator, GeminiServiceError, parse_gemini_error
from src.chunking.semantic_chunking import semantic_chunker
from src.chunking.chunker import text_splitter

import pymupdf
import pymupdf4llm


st.set_page_config(
    page_title="DocuRAG",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

TOP_K = 5
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"

# Stages executed inside build_knowledge_base(). Shown as completed only
# after the function returns successfully.
PIPELINE_STAGES = [
    "Extracting document structure",
    "Converting PDF to markdown",
    "Building document hierarchy",
    "Creating semantic chunks",
    "Generating embeddings",
    "Building FAISS index",
    "Initializing LLM",
]


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
def init_session_state():
    defaults = {
        "messages": [],
        "rag_ready": False,
        "stats": [],
        "vector_store": None,
        "llm": None,
        "document_ids": [],
        "uploader_key": 0,   # bumping this resets the file uploader
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------------------------------------------------
# Backend pipeline (unchanged)
# -------------------------------------------------------------------

def build_knowledge_base(pdf_documents, on_progress=None, existing_store=None):
    """
    Build ONE in-memory FAISS index from multiple PDF documents.

    on_progress: optional callable(stage: str, detail: str = "") used only
    by the UI to report which pipeline stage is running. Pipeline logic is
    unchanged.

    Returns (vector_store, llm, document_stats)
    """

    def notify(stage, detail=""):
        if on_progress is not None:
            on_progress(stage, detail)

    all_embeddings = []
    all_metadata = []
    document_stats = []
    total_docs = len(pdf_documents)

    for doc_number, document in enumerate(pdf_documents, start=1):

        pdf_path = document["path"]
        document_name = document["name"]
        document_id = document["document_id"]

        label = (
            document_name
            if total_docs == 1
            else f"{document_name} ({doc_number} of {total_docs})"
        )

        # 1. Dynamic heading detection
        notify("Analyzing document structure", label)
        header_builder = HeaderInfoBuilder()
        hdr_info = header_builder.build(str(pdf_path))

        # 2. PDF -> Markdown
        notify("Converting pages to markdown", label)
        pymupdf4llm.use_layout(False)

        with pymupdf.open(str(pdf_path)) as doc:
            markdown = pymupdf4llm.to_markdown(
                doc,
                show_progress=False,
                hdr_info=hdr_info,
            )
            pages = doc.page_count

        # 3. Markdown -> hierarchical tree
        notify("Building document hierarchy", label)
        markdown_tree = build_tree(text=markdown)

        # 4. Semantic chunking
        notify("Creating semantic chunks", label)
        chunks = semantic_chunker([markdown_tree])

        # 5. Recursive splitting
        notify("Splitting into passages", label)
        final_chunks = []

        for chunk in chunks:
            splitted_chunks = text_splitter(chunk["content"])

            splitted_chunks = [
                text.strip()
                for text in splitted_chunks
                if text and text.strip()
            ]

            if not splitted_chunks:
                continue

            final_chunks.append({
                "title": chunk["title"],
                "chunks": splitted_chunks,
                "breadcrumb": chunk["breadcrumb"],
            })

        # 6. Prepare embeddings + metadata
        embedding_texts = []
        embedding_metadata = []

        for chunk in final_chunks:
            for text in chunk["chunks"]:
                embedding_texts.append(text)
                embedding_metadata.append({
                    "document_id": document_id,
                    "document_name": document_name,
                    "title": chunk["title"],
                    "breadcrumb": chunk["breadcrumb"],
                    "content": text,
                })

        if not embedding_texts:
            raise ValueError(
                f"No text chunks were produced from "
                f"'{document_name}'. "
                f"Check the extraction/chunking pipeline."
            )

        # 7. Generate embeddings
        notify(
            "Generating embeddings",
            f"{label} · {len(embedding_texts):,} passages",
        )
        embeddings = chunk_encoder(embedding_texts)

        all_embeddings.append(embeddings)
        all_metadata.extend(embedding_metadata)

        document_stats.append({
            "document_id": document_id,
            "document": document_name,
            "size": document["size"],
            "pages": pages,
            "semantic_chunks": len(final_chunks),
            "vector_chunks": len(embedding_texts),
        })

    if not all_embeddings:
        raise ValueError(
            "No embeddings were generated from the uploaded documents."
        )

    import numpy as np

    combined_embeddings = np.vstack(all_embeddings)
    dimension = combined_embeddings.shape[1]

    # Created before touching the index, so a failure here cannot leave
    # vectors in the store that the UI doesn't know about.
    notify("Initializing language model")
    llm = LLMResponseGenerator()

    notify(
        "Storing in FAISS index",
        f"{len(all_metadata):,} passages from {total_docs} "
        f"document{'s' if total_docs != 1 else ''}",
    )

    # Adding sources: extend the existing index. First build: create one.
    vector_store = (
        existing_store
        if existing_store is not None
        else FaissVectorStore(dimension=dimension)
    )
    vector_store.add(
        embeddings=combined_embeddings,
        metadata_list=all_metadata,
    )

    # document_stats covers only the documents processed in this call
    return vector_store, llm, document_stats


def run_query(question: str, vector_store, llm, k: int = 5):
    """
    Embeds the question, retrieves top-k chunks from all uploaded
    documents, and generates an answer grounded in that context.
    """

    query_embedding = chunk_encoder([question])

    results = vector_store.search(
        query_embedding,
        k=k
    )[0]

    context = llm.build_context(results)

    prompt = llm.build_prompt(
        user_question=question,
        context=context
    )

    response = llm.generate_response(
        prompt=prompt
    )

    sources = []

    for r in results:

        sources.append({
            "document_id": r.get("document_id"),
            "document_name": r.get(
                "document_name",
                "Unknown document"
            ),
            "title": r.get(
                "title",
                "Untitled"
            ),
            "breadcrumb": r.get(
                "breadcrumb",
                ""
            ),
            "content": r.get(
                "content",
                ""
            ),
            "score": r.get("distance"),
        })

    return response.text, sources

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

CSS = """
<style>
/* Reading-width column; the chat input inherits the same width. */
.block-container,
[data-testid="stBottomBlockContainer"] { max-width: 760px; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; }

/* Ensure popover overlays are never clipped by parent Streamlit blocks */
.stMarkdown,
[data-testid="stMarkdownContainer"],
.element-container,
[data-testid="stVerticalBlock"] {
    overflow: visible !important;
}

/* Top bar */
.topbar {
    font-size: 0.85rem; opacity: 0.7;
    padding-bottom: 0.6rem; margin-bottom: 1.2rem;
    padding-top: 0.6rem;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}
.topbar b { font-weight: 600; }

/* User question: subtle gray box, left-aligned */
.user-msg {
    background: rgba(128,128,128,0.14);
    border-radius: 14px;
    padding: 0.65rem 1rem;
    margin: 1.6rem 0 0.9rem 0;
    width: fit-content; max-width: 85%;
    line-height: 1.5;
}

/* -------------------------------------------------------------------
   NotebookLM Style Sources Tray & Chips
   ------------------------------------------------------------------- */
.nlm-sources-container {
    margin-top: 1.2rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(128, 128, 128, 0.18);
    position: relative;
    user-select: none;
}

.nlm-sources-header {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.65rem;
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-color, inherit);
    opacity: 0.72;
}

.nlm-sources-header svg {
    width: 14px;
    height: 14px;
    fill: currentColor;
    opacity: 0.85;
}

.nlm-sources-count {
    font-size: 0.7rem;
    opacity: 0.65;
    font-weight: 400;
    text-transform: none;
    letter-spacing: normal;
    background: rgba(128, 128, 128, 0.12);
    padding: 1px 7px;
    border-radius: 999px;
}

.nlm-chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    position: relative;
}

/* Individual Source Chip */
.nlm-chip {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 12px 4px 6px;
    border-radius: 9999px;
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.22);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-color, inherit);
    cursor: pointer;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    outline: none;
}

.nlm-chip:hover,
.nlm-chip:focus,
.nlm-chip:focus-within {
    background: rgba(66, 133, 244, 0.12);
    border-color: rgba(66, 133, 244, 0.45);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.nlm-chip-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(66, 133, 244, 0.18);
    color: #3b82f6;
    font-size: 0.72rem;
    font-weight: 700;
}

.nlm-chip-title {
    max-width: 180px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}

/* -------------------------------------------------------------------
   Inline Citation Pills (in answer prose)
   ------------------------------------------------------------------- */
.nlm-cite-pill {
    position: relative;
    display: inline-flex;
    align-items: center;
    vertical-align: baseline;
    margin: 0 2px;
    cursor: pointer;
    outline: none;
}

.nlm-cite-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 17px;
    height: 17px;
    padding: 0 4px;
    font-size: 0.68rem;
    font-weight: 700;
    border-radius: 6px;
    background: rgba(66, 133, 244, 0.15);
    color: #3b82f6;
    border: 1px solid rgba(66, 133, 244, 0.28);
    transition: all 0.15s ease;
    user-select: none;
}

.nlm-cite-pill:hover .nlm-cite-badge,
.nlm-cite-pill:focus .nlm-cite-badge {
    background: #3b82f6;
    color: #ffffff;
    transform: scale(1.08);
    box-shadow: 0 2px 6px rgba(59, 130, 246, 0.35);
}

/* -------------------------------------------------------------------
   Hover Popover Card (NotebookLM style)
   ------------------------------------------------------------------- */
.nlm-popover {
    position: absolute;
    bottom: calc(100% + 9px);
    left: 0;
    width: 375px;
    max-width: min(375px, 86vw);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transition: opacity 0.18s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.18s cubic-bezier(0.16, 1, 0.3, 1),
                visibility 0.18s;
    transform: translateY(6px);
    z-index: 99999 !important;
    border-radius: 12px;
    background: var(--secondary-background-color, #1f2128);
    border: 1px solid rgba(128, 128, 128, 0.25);
    box-shadow: 0 16px 36px -4px rgba(0, 0, 0, 0.32),
                0 6px 12px -2px rgba(0, 0, 0, 0.12);
    padding: 13px 15px;
    text-align: left;
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--text-color, inherit);
    cursor: default;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

.nlm-popover-right {
    left: auto;
    right: 0;
}

.nlm-popover-inline {
    left: 50%;
    transform: translateX(-50%) translateY(6px);
}

/* Invisible bridge so mouse moving from trigger to popover does not lose hover */
.nlm-popover::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 0;
    width: 100%;
    height: 12px;
    background: transparent;
}

/* Subtle notch / arrow pointer */
.nlm-popover::before {
    content: "";
    position: absolute;
    top: 100%;
    left: 20px;
    border-width: 6px;
    border-style: solid;
    border-color: var(--secondary-background-color, #1f2128) transparent transparent transparent;
}
.nlm-popover-right::before {
    left: auto;
    right: 20px;
}
.nlm-popover-inline::before {
    left: 50%;
    transform: translateX(-50%);
}

/* Popover visibility on hover/focus */
.nlm-chip:hover .nlm-popover,
.nlm-chip:focus .nlm-popover,
.nlm-chip:focus-within .nlm-popover {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateY(0);
}

.nlm-cite-pill:hover .nlm-popover-inline,
.nlm-cite-pill:focus .nlm-popover-inline,
.nlm-cite-pill:focus-within .nlm-popover-inline {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateX(-50%) translateY(0);
}

/* Popover Content Structure */
.nlm-popover-inner {
    display: flex;
    flex-direction: column;
}

.nlm-popover-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 5px;
}

.nlm-popover-badge {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(66, 133, 244, 0.18);
    color: #3b82f6;
}

.nlm-popover-score {
    font-size: 0.72rem;
    opacity: 0.65;
    font-family: monospace;
}

.nlm-popover-title {
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 3px;
    line-height: 1.35;
    color: var(--text-color, inherit);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.nlm-popover-breadcrumb {
    font-size: 0.75rem;
    opacity: 0.65;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.nlm-popover-quote {
    position: relative;
    display: block;
}

.nlm-popover-content {
    display: block;
    background: rgba(128, 128, 128, 0.08);
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 9px 12px;
    font-size: 0.82rem;
    line-height: 1.55;
    max-height: 180px;
    overflow-y: auto;
    white-space: normal;
    word-break: break-word;
    color: var(--text-color, inherit);
}

.nlm-popover-content::-webkit-scrollbar {
    width: 4px;
}
.nlm-popover-content::-webkit-scrollbar-thumb {
    background: rgba(128, 128, 128, 0.3);
    border-radius: 4px;
}

/* -------------------------------------------------------------------
   Sidebar NotebookLM-style Source Card
   ------------------------------------------------------------------- */
.side-title {
    font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; opacity: 0.6; margin-bottom: 0.4rem;
}
.nlm-sidebar-card {
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    margin: 0.4rem 0 0.8rem 0;
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.18);
    transition: all 0.2s ease;
}
.nlm-sidebar-card:hover {
    background: rgba(128, 128, 128, 0.12);
    border-color: rgba(66, 133, 244, 0.35);
}
.nlm-doc-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.45rem;
}
.nlm-doc-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: rgba(66, 133, 244, 0.16);
    color: #3b82f6;
    font-size: 0.95rem;
    flex-shrink: 0;
}
.nlm-doc-info {
    overflow: hidden;
    flex-grow: 1;
}
.nlm-doc-name {
    font-size: 0.88rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}
.nlm-doc-status {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.7rem;
    color: #10b981;
    font-weight: 500;
    margin-top: 1px;
}
.nlm-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
}
.nlm-doc-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 0.4rem;
}
.nlm-meta-chip {
    font-size: 0.7rem;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(128, 128, 128, 0.12);
    opacity: 0.8;
}

/* Empty / ready states */
.center-note { text-align: center; padding: 4.5rem 1rem 0 1rem; }
.center-note .icon { font-size: 2.2rem; }
.center-note .title { font-size: 1.25rem; font-weight: 600; margin: 0.4rem 0 0.3rem 0; }
.center-note .body { opacity: 0.65; line-height: 1.5; }

/* -------------------------------------------------------------------
   NotebookLM Style Gemini Error Alert Card
   ------------------------------------------------------------------- */
.nlm-error-card {
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin: 1.1rem 0 0.8rem 0;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.28);
    color: var(--text-color, inherit);
    position: relative;
    user-select: text;
}
.nlm-error-card.nlm-error-warn {
    background: rgba(245, 158, 11, 0.08);
    border-color: rgba(245, 158, 11, 0.28);
}
.nlm-error-card.nlm-error-info {
    background: rgba(59, 130, 246, 0.08);
    border-color: rgba(59, 130, 246, 0.28);
}
.nlm-error-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.5rem;
}
.nlm-error-icon {
    font-size: 1.35rem;
    line-height: 1;
}
.nlm-error-title-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    flex-grow: 1;
}
.nlm-error-title {
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.3;
}
.nlm-error-badge {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(239, 68, 68, 0.16);
    color: #ef4444;
}
.nlm-error-warn .nlm-error-badge {
    background: rgba(245, 158, 11, 0.18);
    color: #f59e0b;
}
.nlm-error-info .nlm-error-badge {
    background: rgba(59, 130, 246, 0.18);
    color: #3b82f6;
}
.nlm-error-message {
    font-size: 0.86rem;
    line-height: 1.5;
    margin-bottom: 0.55rem;
    opacity: 0.92;
}
.nlm-error-advice {
    font-size: 0.82rem;
    line-height: 1.45;
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    background: rgba(128, 128, 128, 0.09);
    border-left: 3px solid #ef4444;
}
.nlm-error-warn .nlm-error-advice {
    border-left-color: #f59e0b;
}
.nlm-error-info .nlm-error-advice {
    border-left-color: #3b82f6;
}
.nlm-error-advice strong {
    font-weight: 600;
}


.processing-status {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    margin: 32px auto; padding: 14px 20px;
    width: fit-content; max-width: 90%;
    border-radius: 12px;
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.18);
}
.processing-spinner {
    flex-shrink: 0; width: 16px; height: 16px;
    border: 2px solid rgba(128,128,128,0.3);
    border-top-color: currentColor; border-radius: 50%;
    animation: processing-spin 0.8s linear infinite;
}
.processing-text { display: flex; flex-direction: column; gap: 2px; }
.processing-stage {
    font-size: 0.95rem; font-weight: 500;
    animation: processing-pulse 1.6s ease-in-out infinite;
}
.processing-detail { font-size: 0.78rem; opacity: 0.6; word-break: break-word; }
.processing-complete .processing-stage,
.processing-error .processing-stage { animation: none; }
.processing-check { font-size: 16px; }
.processing-error { color: #ef4444; }
@keyframes processing-spin { to { transform: rotate(360deg); } }
@keyframes processing-pulse { 0%, 100% { opacity: 0.55; } 50% { opacity: 1; } }

/* Rendered markdown inside source popovers (inline elements only) */
.nlm-popover-content .md-p,
.nlm-popover-content .md-h,
.nlm-popover-content .md-li,
.nlm-popover-content .md-pre { display: block; }
.nlm-popover-content .md-p  { margin: 0 0 0.5em 0; }
.nlm-popover-content .md-h  { font-weight: 600; margin: 0.2em 0 0.35em 0; }
.nlm-popover-content .md-li { margin: 0 0 0.2em 0.9em; text-indent: -0.9em; }
.nlm-popover-content .md-pre {
    font-family: monospace; font-size: 0.78rem; white-space: pre-wrap;
    background: rgba(128,128,128,0.14); border-radius: 6px;
    padding: 6px 8px; margin: 0.3em 0;
}
.nlm-popover-content .md-row {
    display: flex; gap: 8px; padding: 2px 0;
    border-bottom: 1px solid rgba(128,128,128,0.2); font-size: 0.78rem;
}
.nlm-popover-content .md-row-head { font-weight: 600; }
.nlm-popover-content .md-cell { flex: 1; min-width: 0; word-break: break-word; }
.nlm-popover-content code {
    background: rgba(128,128,128,0.18); border-radius: 4px;
    padding: 0 4px; font-size: 0.78rem;
}

/* Loader inside the sidebar: compact and left-aligned */
section[data-testid="stSidebar"] .processing-status {
    margin: 0.75rem 0;
    padding: 10px 12px;
    justify-content: flex-start;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
}
section[data-testid="stSidebar"] .processing-stage { font-size: 0.85rem; }
</style>
"""


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def get_page_count(pdf_path: Path):
    """UI-only helper: read page count from the temp PDF. None on failure."""
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return None


def md_escape(text: str) -> str:
    """Escape markdown characters so titles render literally in labels."""
    text = " ".join(str(text).split())
    for ch in "\\`*_{}[]<>#|":
        text = text.replace(ch, "\\" + ch)
    return text


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def marker(index: int) -> str:
    return CIRCLED[index - 1] if 1 <= index <= len(CIRCLED) else f"[{index}]"


def fmt_score(score) -> str:
    """Backend value is stored under 'distance', so it is shown as Distance."""
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def render_technical_details(details: str):
    with st.expander("Technical details"):
        st.code(details, language="text")


# -------------------------------------------------------------------
# Build flow
# -------------------------------------------------------------------
def request_build():
    uploaded = st.session_state.get(f"uploader_{st.session_state.uploader_key}")

    if not uploaded:
        return

    existing = {
        (d.get("document"), d.get("size")) for d in st.session_state.stats
    }

    new_documents = [
        {
            "name": file.name,
            "size": file.size,
            "data": file.getvalue(),
            "document_id": str(uuid4()),
        }
        for file in uploaded
        if (file.name, file.size) not in existing
    ]

    if new_documents:
        st.session_state.pending_upload = new_documents


def handle_build(uploads, progress_slot):
    """Adds `uploads` to the knowledge base. Returns number added, or None on failure."""

    temp_paths = []

    def show_progress(stage, detail=""):
        """One line that changes as the pipeline advances."""
        detail_html = (
            f'<div class="processing-detail">{html.escape(detail)}</div>'
            if detail else ""
        )
        progress_slot.markdown(
            '<div class="processing-status">'
            '<div class="processing-spinner"></div>'
            '<div class="processing-text">'
            f'<div class="processing-stage">{html.escape(stage)}…</div>'
            f'{detail_html}'
            '</div></div>',
            unsafe_allow_html=True,
        )

    try:
        pdf_documents = []

        # 1. Read uploaded files into temp files
        for n, upload in enumerate(uploads, start=1):
            label = (
                upload["name"] if len(uploads) == 1
                else f"{upload['name']} ({n} of {len(uploads)})"
            )
            show_progress("Reading document", label)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(upload["data"])
                tmp_path = Path(tmp.name)

            temp_paths.append(tmp_path)
            pdf_documents.append({
                "path": tmp_path,
                "name": upload["name"],
                "size": upload["size"],
                "document_id": upload["document_id"],
            })

        # 2. Build, extending the existing index if there is one
        already_ready = st.session_state.rag_ready
        vector_store, llm, new_stats = build_knowledge_base(
            pdf_documents,
            on_progress=show_progress,
            existing_store=st.session_state.vector_store if already_ready else None,
        )

        # 3. Save to session state (append, never replace)
        st.session_state.vector_store = vector_store
        st.session_state.llm = llm
        st.session_state.rag_ready = True
        if not already_ready:
            st.session_state.messages = []  # keep the chat when adding sources
        st.session_state.stats = st.session_state.stats + new_stats
        st.session_state.document_ids = (
            st.session_state.document_ids + [d["document_id"] for d in new_stats]
        )
        return len(new_stats)

    except Exception as exc:
        progress_slot.markdown(
            '<div class="processing-status processing-error">'
            '<div>⚠</div>'
            '<div class="processing-text">'
            '<div class="processing-stage">Could not prepare sources</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.error("Something went wrong while preparing the sources.")
        render_technical_details(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        return None

    finally:
        for tmp_path in temp_paths:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
# -------------------------------------------------------------------
# Sidebar: sources
# -------------------------------------------------------------------
def render_source_list():

    st.markdown(
        '<div class="side-title">Your sources</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.rag_ready:
        st.caption("No sources yet.")
        return

    documents = st.session_state.stats

    for stats in documents:

        doc_name = stats.get(
            "document",
            "Unknown"
        )

        chips = []

        if stats.get("pages"):
            chips.append(
                f"{stats['pages']} pages"
            )

        if stats.get("size"):
            chips.append(
                format_size(stats["size"])
            )

        if stats.get("vector_chunks"):
            chips.append(
                f"{stats['vector_chunks']:,} chunks"
            )

        chips_html = "".join(
            f'<span class="nlm-meta-chip">'
            f'{html.escape(c)}'
            f'</span>'
            for c in chips
        )

        st.markdown(
            f"""
            <div class="nlm-sidebar-card">
                <div class="nlm-doc-header">
                    <div class="nlm-doc-icon">
                        📄
                    </div>
                    <div class="nlm-doc-info">
                        <div
                            class="nlm-doc-name"
                            title="{html.escape(doc_name)}"
                        >
                            {html.escape(doc_name)}
                        </div>
                        <div class="nlm-doc-status">
                            <span class="nlm-status-dot"></span>
                            <span>Active Source</span>
                        </div>
                    </div>
                </div>
                <div class="nlm-doc-chips">
                    {chips_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_sidebar():

    # Read one-shot flags first so widgets can be disabled during a build.
    pending = st.session_state.pop("pending_upload", None)
    notice = st.session_state.pop("build_notice", None)
    building = pending is not None

    with st.sidebar:

        st.markdown('<div class="side-title">Sources</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "+ Add sources",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="Upload one or more PDFs to add them to the knowledge base.",
            disabled=building,
        )

        # Split the selection into genuinely new files and ones already added.
        active_keys = {
            (d.get("document"), d.get("size")) for d in st.session_state.stats
        }
        selected = uploaded or []
        new_files = [f for f in selected if (f.name, f.size) not in active_keys]
        already_added = [f for f in selected if (f.name, f.size) in active_keys]

        if new_files:
            st.caption(f"{len(new_files)} new PDF(s) selected")
            for file in new_files:
                st.caption(f"• {file.name} · {format_size(file.size)}")
            if already_added:
                st.caption(f"{len(already_added)} already added, skipped")

            st.button(
                "Add sources" if st.session_state.rag_ready else "Prepare sources",
                type="primary",
                width="stretch",
                on_click=request_build,
                disabled=building,
            )
        elif already_added:
            st.caption("Already in your sources.")

        # Loader lives directly under the button.
        progress_slot = st.empty()

        if building:
            added = handle_build(pending, progress_slot)
            if added is not None:
                # New key -> empty uploader on the next run.
                st.session_state.uploader_key += 1
                st.session_state.build_notice = added
                st.rerun()
        elif notice:
            progress_slot.markdown(
                '<div class="processing-status processing-complete">'
                '<div class="processing-check">✓</div>'
                '<div class="processing-text">'
                f'<div class="processing-stage">{notice} source{"s" if notice != 1 else ""} added</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        render_source_list()

        if st.session_state.rag_ready:
            st.caption("All sources are searched together as one knowledge base.")
        else:
            st.caption("Upload one or more PDFs to begin.")

        st.divider()
        st.caption("PDF → Semantic Chunking → Embeddings → FAISS → LLM")

        
# -------------------------------------------------------------------
# Main area
# -------------------------------------------------------------------
def render_topbar():

    documents = st.session_state.stats

    if documents:

        if len(documents) == 1:

            suffix = (
                f" &nbsp;/&nbsp; "
                f"{html.escape(documents[0]['document'])}"
            )

        else:

            suffix = (
                f" &nbsp;/&nbsp; "
                f"{len(documents)} sources"
            )

    else:

        suffix = ""

    st.markdown(
        f'<div class="topbar">'
        f'<b>DocuRAG</b>{suffix}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="center-note">
            <div class="icon">📚</div>
            <div class="title">
                Start with your sources
            </div>
            <div class="body">
                Upload one or more PDFs to create
                your document knowledge base.<br>
                Use <b>+ Add sources</b> in the sidebar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ready_state():

    documents = st.session_state.stats

    if len(documents) == 1:

        name = html.escape(
            documents[0].get(
                "document",
                "your source"
            )
        )

        body = f"{name} is ready."

    else:

        body = (
            f"{len(documents)} documents are ready. "
            f"You can ask questions across all of them."
        )

    st.markdown(
        f"""
        <div class="center-note">
            <div class="title">
                Ask questions about your sources
            </div>
            <div class="body">
                {body}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Markdown -> inline-only HTML (for source popovers)
# -------------------------------------------------------------------

_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

# Entities keep leftover symbols from being re-parsed by st.markdown.
_MD_ENTITIES = str.maketrans({
    "*": "&#42;", "_": "&#95;", "`": "&#96;",
    "[": "&#91;", "]": "&#93;", "$": "&#36;", "\\": "&#92;",
})


def plain_text(text) -> str:
    """Strip markdown symbols from short labels (titles, breadcrumbs)."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", str(text))
    text = re.sub(r"^\s*#{1,6}\s*", "", text)
    text = re.sub(r"\*+|`+|~~", "", text)
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)
    return " ".join(text.split())


def _md_inline(s: str) -> str:
    """Inline markdown -> inline HTML. `s` must already be HTML-escaped."""
    s = s.replace("&lt;br&gt;", " ").replace("&lt;br/&gt;", " ")
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"__(.+?)__", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"(?<![\w_])_(?![\s_])(.+?)(?<![\s_])_(?![\w_])", r"<em>\1</em>", s)
    return s


def _table_cells(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def markdown_to_html(text: str) -> str:
    """
    Converts chunk markdown to a single-line string of INLINE elements
    (span/strong/em/code). No block tags and no blank lines, so it is safe
    inside the popover spans and inside st.markdown paragraphs.
    """
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out, para = [], []
    i = 0

    def flush():
        if para:
            body = _md_inline(html.escape(" ".join(para)))
            out.append(f'<span class="md-p">{body}</span>')
            para.clear()

    while i < len(lines):
        line = lines[i].strip()

        # fenced code block
        if line.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            escaped = html.escape("\n".join(code)).replace("\n", "&#10;")
            out.append(f'<span class="md-pre">{escaped}</span>')
            i += 1
            continue

        if not line:
            flush(); i += 1; continue

        if re.fullmatch(r"[-*_]{3,}", line):  # horizontal rule
            flush(); i += 1; continue

        # table: header row followed by a separator row
        if line.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [_table_cells(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_table_cells(lines[i]))
                i += 1
            for r, row in enumerate(rows):
                cls = "md-row md-row-head" if r == 0 else "md-row"
                cells = "".join(
                    f'<span class="md-cell">{_md_inline(html.escape(c))}</span>'
                    for c in row
                )
                out.append(f'<span class="{cls}">{cells}</span>')
            continue

        m = re.match(r"#{1,6}\s+(.*)", line)
        if m:
            flush()
            out.append(f'<span class="md-h">{_md_inline(html.escape(m.group(1).strip("# ")))}</span>')
            i += 1
            continue

        m = re.match(r"[-*+•]\s+(.*)", line)
        if m:
            flush()
            out.append(f'<span class="md-li">• {_md_inline(html.escape(m.group(1)))}</span>')
            i += 1
            continue

        m = re.match(r"(\d+)[.)]\s+(.*)", line)
        if m:
            flush()
            out.append(f'<span class="md-li">{m.group(1)}. {_md_inline(html.escape(m.group(2)))}</span>')
            i += 1
            continue

        if line.startswith(">"):
            line = line.lstrip("> ").strip()

        para.append(line)
        i += 1

    flush()
    return "".join(out).translate(_MD_ENTITIES)


def build_source_popover_card(
    idx: int,
    source: dict,
    is_right_aligned: bool = False,
    is_inline: bool = False,
) -> str:
    """Floating popover with metadata and the retrieved chunk, markdown rendered."""
    title = html.escape(plain_text(source.get("title") or f"Source {idx}")).translate(_MD_ENTITIES)
    breadcrumb = html.escape(plain_text(source.get("breadcrumb") or "")).translate(_MD_ENTITIES)
    content = markdown_to_html(source.get("content") or "")
    score = source.get("score")

    score_html = ""
    if score is not None:
        try:
            score_val = f"{float(score):.3f}"
        except Exception:
            score_val = str(score)
        score_html = f'<span class="nlm-popover-score">Dist: {score_val}</span>'

    bc_html = ""
    if breadcrumb:
        bc_html = f'<span class="nlm-popover-breadcrumb"><span style="opacity:0.8;">📍</span> {breadcrumb}</span>'

    align_class = "nlm-popover-inline" if is_inline else ("nlm-popover-right" if is_right_aligned else "")

    return (
        f'<span class="nlm-popover {align_class}">'
        f'<span class="nlm-popover-inner">'
        f'<span class="nlm-popover-header">'
        f'<span class="nlm-popover-badge">Source {idx}</span>{score_html}'
        f'</span>'
        f'<span class="nlm-popover-title">{title}</span>'
        f'{bc_html}'
        f'<span class="nlm-popover-quote"><span class="nlm-popover-content">{content}</span></span>'
        f'</span></span>'
    )

def build_sources_tray_html(sources: list) -> str:
    """Builds the modern horizontal NotebookLM source chips tray."""
    if not sources:
        return ""

    chips_html = []
    total = len(sources)
    for i, src in enumerate(sources, start=1):
        # raw_title = src.get("title") or f"Source {i}"
        # title = html.escape(raw_title)
        # display_title = title if len(title) <= 24 else title[:23] + "…"
        clean_title = plain_text(src.get("title") or f"Source {i}")
        title = html.escape(clean_title)
        display_title = html.escape(shorten(clean_title, 24))
        is_right = i > (total // 2)
        popover = build_source_popover_card(i, src, is_right_aligned=is_right, is_inline=False)

        chip = f"""<span class="nlm-chip" tabindex="0" role="button" aria-label="Source {i}: {title}">
            <span class="nlm-chip-num">{i}</span>
            <span class="nlm-chip-title">{display_title}</span>
            {popover}
        </span>"""
        chips_html.append(chip)

    chips_str = "\n".join(chips_html)
    count_str = f"{total} source{'s' if total != 1 else ''}"

    return f"""<div class="nlm-sources-container">
        <div class="nlm-sources-header">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
            </svg>
            <span>Sources</span>
            <span class="nlm-sources-count">{count_str}</span>
        </div>
        <div class="nlm-chips-row">
            {chips_str}
        </div>
    </div>"""


def replace_citations_in_text(text: str, sources: list) -> str:
    """
    Finds citation references in the generated text (e.g. [Source 1], [1], [Source 1, 2])
    and turns them into interactive hoverable NotebookLM citation badges with popovers.
    Code blocks and inline code are left untouched.
    """
    if not sources or not text:
        return text

    pattern_code = r"(```[\s\S]*?```|`[^`\n]*?`)"
    segments = re.split(pattern_code, text)

    def make_citation_pill(num_str: str) -> str:
        try:
            num = int(num_str)
        except ValueError:
            return num_str
        if 1 <= num <= len(sources):
            src = sources[num - 1]
            popover = build_source_popover_card(num, src, is_inline=True)
            return (
                f'<span class="nlm-cite-pill" tabindex="0" role="button" aria-label="Source {num} citation">'
                f'<span class="nlm-cite-badge">{num}</span>'
                f'{popover}'
                f'</span>'
            )
        return f"[{num_str}]"

    def replace_citation_group(match):
        group_content = match.group(1)
        numbers = re.findall(r"\b(\d+)\b", group_content)
        if not numbers:
            return match.group(0)
        return "".join([make_citation_pill(n) for n in numbers])

    for i in range(len(segments)):
        if i % 2 == 0:
            segments[i] = re.sub(
                r"\[\s*(?:Source\s*)?(\d+(?:\s*,\s*(?:Source\s*)?\d+)*)\s*\](?!\()",
                replace_citation_group,
                segments[i]
            )

    return "".join(segments)


def render_sources(sources):
    """Renders modern NotebookLM-style interactive source chips with hover popovers."""
    tray_html = build_sources_tray_html(sources)
    if tray_html:
        st.markdown(tray_html, unsafe_allow_html=True)


def render_user_message(text: str):
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="user-msg">{safe}</div>', unsafe_allow_html=True)


def render_error_card(error_data: dict, msg_idx: int = 0):
    """Renders a modern NotebookLM-style error alert with status badge and actionable advice."""
    category = error_data.get("category", "UNKNOWN")
    title = html.escape(error_data.get("title", "Gemini API Error"))
    message = html.escape(error_data.get("message", "An error occurred while communicating with Gemini."))
    advice = html.escape(error_data.get("advice", "Please try again."))
    code = error_data.get("code")
    retryable = error_data.get("retryable", False)
    question = error_data.get("question", "")

    CATEGORY_META = {
        "HIGH_DEMAND": ("⚡", f"High Demand ({code or 503})", "nlm-error-warn"),
        "QUOTA_EXCEEDED": ("🛑", f"Quota Reached ({code or 429})", "nlm-error-card"),
        "RATE_LIMIT": ("⏳", f"Rate Limit ({code or 429})", "nlm-error-warn"),
        "TOKEN_LIMIT": ("📏", f"Context Exceeded ({code or 400})", "nlm-error-warn"),
        "PERMISSION_DENIED": ("🔒", f"Auth Error ({code or 403})", "nlm-error-card"),
        "REGION_UNSUPPORTED": ("🌐", f"Region Restricted ({code or 403})", "nlm-error-card"),
        "SAFETY_FILTER": ("🛡️", "Safety Filter", "nlm-error-warn"),
        "RECITATION": ("📄", "Copyright Policy", "nlm-error-warn"),
        "MODEL_NOT_FOUND": ("🔍", f"Model Not Found ({code or 404})", "nlm-error-warn"),
        "SERVER_ERROR": ("⚠️", f"Google Server Error ({code or 500})", "nlm-error-warn"),
        "NETWORK_ERROR": ("📶", "Connection Error", "nlm-error-info"),
        "EMPTY_RESPONSE": ("💬", "No Response", "nlm-error-warn"),
        "UNKNOWN": ("⚠️", f"API Error ({code})" if code else "API Error", "nlm-error-card"),
    }

    icon, badge, card_cls = CATEGORY_META.get(category, ("⚠️", "API Error", "nlm-error-card"))

    card_html = f"""
    <div class="nlm-error-card {card_cls}">
        <div class="nlm-error-header">
            <span class="nlm-error-icon">{icon}</span>
            <div class="nlm-error-title-wrap">
                <span class="nlm-error-title">{title}</span>
                <span class="nlm-error-badge">{badge}</span>
            </div>
        </div>
        <div class="nlm-error-message">{message}</div>
        <div class="nlm-error-advice">
            <strong>Actionable Tip:</strong> {advice}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    tech = error_data.get("technical_details")
    if retryable and question:
        btn_key = f"retry_btn_{msg_idx}_{abs(hash(question)) % 1000000}"
        if st.button("🔄 Retry question", key=btn_key):
            with st.spinner("Retrying with Gemini…"):
                new_reply = answer_question(question)
                if 0 <= msg_idx < len(st.session_state.messages):
                    st.session_state.messages[msg_idx] = new_reply
            st.rerun()

    if tech:
        with st.expander("Technical details & error log"):
            st.code(tech, language="text")


def render_assistant_message(message: dict, msg_idx: int = 0):
    """Answer with inline NotebookLM citations and interactive source chips on hover."""
    if message.get("error_data"):
        render_error_card(message["error_data"], msg_idx=msg_idx)
        return

    if message.get("error"):
        parsed = parse_gemini_error(Exception(message["error"]))
        err_data = {
            "category": parsed.category,
            "title": parsed.title,
            "message": parsed.message,
            "advice": parsed.advice,
            "code": parsed.code,
            "retryable": parsed.retryable,
            "technical_details": message["error"],
            "question": "",
        }
        render_error_card(err_data, msg_idx=msg_idx)
        return

    sources = message.get("sources", [])
    formatted_content = replace_citations_in_text(message.get("content", ""), sources)
    st.markdown(formatted_content, unsafe_allow_html=True)
    render_sources(sources)


def render_chat_message(message: dict, msg_idx: int = 0):
    if message["role"] == "user":
        render_user_message(message["content"])
    else:
        render_assistant_message(message, msg_idx=msg_idx)


def answer_question(question: str) -> dict:
    """Calls run_query() and packages the result or structured error as a message."""
    try:
        answer, sources = run_query(
            question,
            st.session_state.vector_store,
            st.session_state.llm,
            k=TOP_K,
        )
        return {"role": "assistant", "content": answer, "sources": sources}
    except GeminiServiceError as gerr:
        return {
            "role": "assistant",
            "content": "",
            "sources": [],
            "error_data": {
                "category": gerr.category,
                "title": gerr.title,
                "message": gerr.message,
                "advice": gerr.advice,
                "code": gerr.code,
                "retryable": gerr.retryable,
                "technical_details": gerr.technical_details,
                "question": question,
            },
        }
    except Exception as exc:
        parsed = parse_gemini_error(exc)
        return {
            "role": "assistant",
            "content": "",
            "sources": [],
            "error_data": {
                "category": parsed.category,
                "title": parsed.title,
                "message": parsed.message,
                "advice": parsed.advice,
                "code": parsed.code,
                "retryable": parsed.retryable,
                "technical_details": f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                "question": question,
            },
        }


def render_conversation(question):
    if not st.session_state.rag_ready:
        render_empty_state()
        return

    if not st.session_state.messages and not question:
        render_ready_state()

    for idx, message in enumerate(st.session_state.messages):
        render_chat_message(message, msg_idx=idx)

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_chat_message(user_message, msg_idx=len(st.session_state.messages) - 1)

        with st.spinner("Searching your source…"):
            reply = answer_question(question)
        st.session_state.messages.append(reply)
        render_chat_message(reply, msg_idx=len(st.session_state.messages) - 1)


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
def main():

    init_session_state()
    st.markdown(CSS, unsafe_allow_html=True)

    topbar_slot = st.container()

    # Sidebar runs any pending build first (loader shows under the button),
    # so everything below renders with up-to-date state.
    render_sidebar()

    with topbar_slot:
        render_topbar()

    question = st.chat_input(
        "Ask about your sources…",
        disabled=not st.session_state.rag_ready,
    )

    render_conversation(question)



main()