from src.common.common_functions import get_page_lines

# def identify_boilerplate(
#     similarity_df,
#     mean_threshold=90,
#     std_threshold=10
# ):
#     """
#     Identify repeated boilerplate lines using similarity statistics.

#     Parameters
#     ----------
#     similarity_df : pandas.DataFrame
#         DataFrame containing:
#             region
#             line
#             score

#     mean_threshold : float
#         Minimum average similarity required.

#     std_threshold : float
#         Maximum allowed standard deviation.

#     Returns
#     -------
#     pandas.DataFrame
#         Summary statistics and classification.
#     """

#     statistics = (
#         similarity_df
#         .groupby(["region", "line"])["score"]
#         .agg(
#             mean_score="mean",
#             std_score="std"
#         )
#         .reset_index()
#     )

#     statistics["is_boilerplate"] = (
#         (statistics["mean_score"] >= mean_threshold)
#         &
#         (statistics["std_score"] <= std_threshold)
#     )

#     return statistics


# =========================
# 6. Statistical Boilerplate Detection
# =========================

def identify_boilerplate(
    similarity_df,
    mean_threshold=90,
    std_threshold=10
):
    """
    Identify repeated boilerplate lines using similarity statistics.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        DataFrame containing:
            region
            line
            score

    mean_threshold : float
        Minimum average similarity required.

    std_threshold : float
        Maximum allowed standard deviation.

    Returns
    -------
    pandas.DataFrame
        Summary statistics and classification.
    """

    statistics = (
        similarity_df
        .groupby(["region", "line"])["score"]
        .agg(
            mean_score="mean",
            std_score="std"
        )
        .reset_index()
    )

    statistics["is_boilerplate"] = (
        (statistics["mean_score"] >= mean_threshold)
        &
        (statistics["std_score"] <= std_threshold)
    )

    return statistics








# def remove_boilerplate(pdf, boilerplate_config):
#     cleaned_pdf = []

#     for page in pdf:

#         # Keep only non-empty lines
#         page_lines = [
#             line.strip()
#             for line in page.splitlines()
#             if line.strip()
#         ]

#         # Identify lines to remove
#         lines_to_remove = set()

#         for _, row in boilerplate_config.iterrows():

#             region = row["region"]
#             line_number = row["line"]

#             if region == "header":

#                 index = line_number - 1

#             elif region == "footer":

#                 index = len(page_lines) - line_number

#             else:
#                 continue

#             # Make sure the index exists on this page
#             if 0 <= index < len(page_lines):
#                 lines_to_remove.add(index)

#         # Keep all lines except detected boilerplate
#         cleaned_page_lines = [
#             line
#             for index, line in enumerate(page_lines)
#             if index not in lines_to_remove
#         ]

#         cleaned_page = "\n".join(cleaned_page_lines)

#         cleaned_pdf.append(cleaned_page)

#     return cleaned_pdf


# =========================
# 9. Remove Boilerplate
# =========================

def remove_boilerplate(
    pdf,
    boilerplate_config
):

    cleaned_pdf = []

    boilerplate_positions = {
        (row["region"], row["line"])
        for _, row in boilerplate_config.iterrows()
    }

    for page in pdf:

        page_lines = get_page_lines(page)

        lines_to_remove = set()

        for region, line_number in boilerplate_positions:

            if region == "header":

                index = line_number - 1

            elif region == "footer":

                index = len(page_lines) - line_number

            else:

                continue

            if 0 <= index < len(page_lines):

                lines_to_remove.add(index)

        cleaned_page = "\n".join(
            line
            for index, line in enumerate(page_lines)
            if index not in lines_to_remove
        )

        cleaned_pdf.append(cleaned_page)

    return cleaned_pdf