path_stack = []
chunks = []

# Traverse the Tree
def semantic_chunker(tree)-> list[dict]:

    for node in tree:

        # Building the Path Stack
        path_stack.append(node["text"])

        has_children = bool(node["children"])

        if node.get("content") is not None:

            # Produce breadcrumb
            bread_crumb = path_stack.copy()

            content = node.get("content", "")

            # Node-level content
            if has_children:

                subtopics = [
                    child["text"]
                    for child in node["children"]
                ]

                chunk_content = (
                    f"[Content]: {content}\n"
                    f"[Subtopics]: {subtopics}"
                )

            # Leaf-level content
            else:
                chunk_content = content

            # Create chunk
            chunks.append({
                "title": node.get("text", ""),
                "content": chunk_content,
                "breadcrumb": bread_crumb
            })

        # Traverse children
        if has_children:
            semantic_chunker(node["children"])

        # Leave current node
        path_stack.pop()
    return chunks