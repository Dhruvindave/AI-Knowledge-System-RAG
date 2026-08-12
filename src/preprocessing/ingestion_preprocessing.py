from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF
import matplotlib.pyplot as plt
import pandas as pd

BOLD_FLAG = 1 << 4    # bit 4
ITALIC_FLAG = 1 << 1  # bit 1


class HeaderInfoBuilder:

    def __init__(
        self,
        min_occurrence_ratio=0.001,
        min_gap_ratio=1.05,
        max_heading_levels=6,
        min_bold_ratio=None,
    ):
        self.min_occurrence_ratio = min_occurrence_ratio
        self.min_gap_ratio = min_gap_ratio
        self.max_heading_levels = max_heading_levels
        self.min_bold_ratio = min_bold_ratio

    def _extract_font_spans(self, pdf_path: str | Path) -> pd.DataFrame:
        """
        Walk every text span in the PDF and count occurrences grouped by
        (font_size, font_name, bold, italic).

        Returns
        -------
        DataFrame with columns: font_size, font_name, bold, italic, count
        """
        doc = pymupdf.open(pdf_path)
        counter: Counter[tuple[float, str, bool, bool]] = Counter()

        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:  # skip image blocks
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        key = (
                            round(span["size"], 2),
                            span["font"],
                            bool(span["flags"] & BOLD_FLAG),
                            bool(span["flags"] & ITALIC_FLAG),
                        )
                        counter[key] += 1

        doc.close()

        return pd.DataFrame(
            [
                {
                    "font_size": size,
                    "font_name": font,
                    "bold": bold,
                    "italic": italic,
                    "count": count,
                }
                for (size, font, bold, italic), count in counter.items()
            ]
        )

    def _summarize_by_size(self, span_df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse span-level stats down to one row per font size:
        total occurrence count, bold/italic usage ratio, font-name diversity,
        and the size gap/ratio down to the next-smaller size (useful for
        detecting natural heading-level cluster boundaries).
        """
        weighted = span_df.assign(
            bold_weighted=span_df["count"] * span_df["bold"],
            italic_weighted=span_df["count"] * span_df["italic"],
        )

        size_df = (
            weighted.groupby("font_size")
            .agg(
                count=("count", "sum"),
                bold_count=("bold_weighted", "sum"),
                italic_count=("italic_weighted", "sum"),
                font_variants=("font_name", "nunique"),
            )
            .reset_index()
            .sort_values("font_size", ascending=False)
            .reset_index(drop=True)
        )

        size_df["bold_ratio"] = size_df["bold_count"] / size_df["count"]
        size_df["italic_ratio"] = size_df["italic_count"] / size_df["count"]
        size_df["gap_below"] = size_df["font_size"] - \
            size_df["font_size"].shift(-1)
        size_df["ratio_below"] = size_df["font_size"] / \
            size_df["font_size"].shift(-1)

        return size_df

    def _build_dynamic_hdr_info(
        self,
        size_df,
        # candidate must be ≥0.1% of all spans to not be an outlier
        min_occurrence_ratio=0.001,
        min_gap_ratio=1.05,           # ≥5% size jump = new heading level
        max_heading_levels=6,
        min_bold_ratio=None,          # optional: e.g. 0.5 to require majority-bold styling
    ):
        """
        Derives heading-level clusters from font-size statistics and returns
        a hdr_info(span, page=None) callable for pymupdf4llm.to_markdown().
        """
        df = size_df.copy().sort_values("font_size", ascending=False).reset_index(drop=True)
        total_count = df["count"].sum()

        # Step 1: body text = size with the highest total occurrence.
        body_size = df.loc[df["count"].idxmax(), "font_size"]

        # Step 2: candidates must be visually larger than body text.
        candidates = df[df["font_size"] > body_size].copy()

        # Step 3: drop outliers -- rare sizes aren't deliberate heading styles.
        min_count = max(2, total_count * min_occurrence_ratio)
        candidates = candidates[candidates["count"] >= min_count]

        # Optional: require the size to be predominantly bold to count as a heading.
        if min_bold_ratio is not None:
            candidates = candidates[candidates["bold_ratio"] >= min_bold_ratio]

        if candidates.empty:
            return lambda span, page=None: ""

        # Step 4: recompute ratio_below on the FILTERED set (neighbors changed
        # after outlier removal), then cluster on dynamic gap thresholds.
        candidates = candidates.sort_values(
            "font_size", ascending=False).reset_index(drop=True)
        candidates["ratio_below"] = candidates["font_size"] / \
            candidates["font_size"].shift(-1)

        cluster_id = 0
        cluster_ids = [0]
        for i in range(1, len(candidates)):
            prev_ratio = candidates.loc[i - 1, "ratio_below"]
            if pd.isna(prev_ratio) or prev_ratio >= min_gap_ratio:
                cluster_id += 1
            cluster_ids.append(cluster_id)
        candidates["cluster"] = cluster_ids

        # Step 5: one markdown level per cluster, largest size = level 1.
        cluster_order = list(dict.fromkeys(candidates["cluster"]))[
            :max_heading_levels]
        cluster_to_level = {c: i + 1 for i, c in enumerate(cluster_order)}

        size_to_level = {
            round(row["font_size"], 1): cluster_to_level[row["cluster"]]
            for _, row in candidates.iterrows()
            if row["cluster"] in cluster_to_level
        }
        sorted_sizes = sorted(size_to_level.keys(), reverse=True)

        def hdr_info(span, page=None):
            size = round(span["size"], 1)
            for s in sorted_sizes:
                if size >= s - 0.3:   # tolerance for float rendering noise
                    return "#" * size_to_level[s] + " "
            return ""

        return hdr_info

    def build(self, pdf_path):
        span_df = self._extract_font_spans(pdf_path)

        size_df = self._summarize_by_size(span_df)

        hdr_info = self._build_dynamic_hdr_info(size_df)

        return hdr_info
