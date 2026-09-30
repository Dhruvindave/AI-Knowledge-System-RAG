from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import CHUNK_SIZE, CHUNK_OVERLAP

def text_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )

    all_splits = text_splitter.split_text(text)

    print(f"Split PDF text into {len(all_splits)} sub-documents.")

    return all_splits
