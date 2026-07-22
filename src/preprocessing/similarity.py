
import rapidfuzz as fuzz

# =========================
# 4. Compute Line Similarity
# =========================


def compute_line_similarity(pdf, candidate_lines=3):
    """
    Compare the first and last `n_lines` of consecutive PDF pages.

    Parameters
    ----------
    pdf : list[str]
        List containing page-wise extracted text.

    n_lines : int
        Number of header/footer candidate lines.

    similarity_fn : function
        RapidFuzz similarity function.
        Examples:
            fuzz.ratio
            fuzz.partial_ratio
            fuzz.token_sort_ratio
            fuzz.token_set_ratio
            fuzz.WRatio

    Returns
    -------
    list[dict]
        Similarity information for every consecutive page pair.
    """

    similarity_records = []

    for page_number, (current_page, next_page) in enumerate(
        zip(pdf, pdf[1:]),
        start=1
    ):

        current_lines = get_page_lines(current_page)
        next_lines = get_page_lines(next_page)

        # Candidate header lines
        current_header = current_lines[:candidate_lines]
        next_header = next_lines[:candidate_lines]

        # Candidate footer lines
        current_footer = current_lines[-candidate_lines:]
        next_footer = next_lines[-candidate_lines:]

        # Compare header lines
        for line_number, (text1, text2) in enumerate(
            zip(current_header, next_header),
            start=1
        ):

            similarity_records.append({
                "page_from": page_number,
                "page_to": page_number + 1,
                "text1": text1,
                "text2": text2,
                "region": "header",
                "line": line_number,
                "score": fuzz.ratio(text1, text2)
            })

        # Compare footer lines
        for line_number, (text1, text2) in enumerate(
            zip(current_footer, next_footer),
            start=1
        ):

            similarity_records.append({
                "page_from": page_number,
                "page_to": page_number + 1,
                "text1": text1,
                "text2": text2,
                "region": "footer",
                "line": line_number,
                "score": fuzz.ratio(text1, text2)
            })

    return pd.DataFrame(similarity_records)
