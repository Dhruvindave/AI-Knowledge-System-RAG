import re


def _normalize_text(text: str) -> str:
    """ 
    Cleans extracted PDF text for better chunking and retrieval. 
    """
    # Step 1: Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # # Step 2: Remove repeated book title/header
    # text = re.sub("Basics of Mathematics\s+\d+\s*", "", text)

    # # Step 3: Remove standalone page numbers
    # text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Step 4: Replace multiple space with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Step 5: Replace multiple blank lines with at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def normalize_pdf(pdf: list[str]) -> list[str]:
    normalized_pdf = []
    i = 0

    for page in pdf:
        normalized_page = _normalize_text(page)
        normalized_pdf.append(normalized_page)

    # Comparing The cleaned PDF vs Original PDF
    for i in range(0, 5):
        print("---------------------------------------")
        print("------------- Original PDF ------------")
        print(pdf[i])
        print("---------------------------------------\n")

        print("---------------------------------------")
        print("------------- Cleaned PDF ------------")
        print(normalized_pdf[i])
        print("---------------------------------------\n")

    return normalized_pdf
