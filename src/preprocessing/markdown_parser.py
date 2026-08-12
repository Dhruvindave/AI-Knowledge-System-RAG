import re

def build_tree(text: str) -> dict:
    """
    Builds a semantic tree from Markdown.

    `depth` represents the node's depth in the semantic tree.

    `heading_level` represents the original Markdown heading level.

    Example:

        # A
        ### B
        ## C
        ### D

    becomes:

        A (depth=0, heading_level=1)
        ├── B (depth=1, heading_level=3)
        └── C (depth=1, heading_level=2)
            └── D (depth=2, heading_level=3)
    """

    root = {
        "text": "",
        "depth": 0,
        "children": [],
    }

    heading_pattern = re.compile(
        r"^(#{1,6})[ \t]+(.+?)\s*$",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(text))

    # No headings
    if not matches:
        root["text"] = text.strip()
        return root

    # Text before the first heading
    preamble = text[:matches[0].start()].strip()

    if preamble:
        root["text"] = preamble

    # Stack contains the current semantic path.
    #
    # Example:
    #
    # [
    #     {"level": 1, "node": ...},
    #     {"level": 2, "node": ...},
    # ]
    #
    stack = []

    for index, match in enumerate(matches):

        heading_marker = match.group(1)
        heading_text = match.group(2).strip()

        # Actual Markdown heading level.
        heading_level = len(heading_marker)

        # Content belongs to this heading until
        # the next heading.
        content_start = match.end()

        if index + 1 < len(matches):
            content_end = matches[index + 1].start()
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()

        # Find nearest previous heading with a
        # smaller Markdown heading level.
        while stack and stack[-1]["level"] >= heading_level:
            stack.pop()

        # Parent is either:
        #
        # 1. nearest smaller heading
        # 2. root if no such heading exists
        if stack:
            parent = stack[-1]["node"]
            depth = parent["depth"] + 1
        else:
            parent = root
            depth = 0

        node = {
            "text": heading_text,
            "depth": depth,
            "heading_level": heading_level,
            "children": [],
        }

        if content:
            node["content"] = content

        parent["children"].append(node)

        # Current node becomes the active heading.
        stack.append({
            "level": heading_level,
            "node": node,
        })

    return root
