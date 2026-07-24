
# =========================
# 3. Normalize Page Lines
# =========================

def get_page_lines(page):

    return [
        line.strip()
        for line in page.splitlines()
        if line.strip()
    ]
