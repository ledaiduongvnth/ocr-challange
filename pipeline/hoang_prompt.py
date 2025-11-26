ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
]

ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.

* Formatting: Maintain consistent formatting with the image, including spacing, indentation, list markers, subscripts/superscripts, and special characters.

* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.

* Forms: Mark checkboxes and radio buttons properly using <input> with the appropriate type and checked attributes.

* Text: Join lines together properly into paragraphs using <p>...</p> tags. Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.

* Layout: Use the simplest possible HTML structure that accurately represents the content of the block.

* Reading order: Make sure the text is accurate and easy for a human to read and interpret. Reading order should be correct and natural from top to bottom, left to right.

* Headers and footers: In header or footer content, lines can easily get stuck together. Remember to separate them into distinct elements when appropriate.
""".strip()


OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks. Each layout block should be a <div> with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format. Bboxes are normalized 0-{{bbox_scale}}. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure

For any region that visually represents a table (with or without borders, including financial statements, schedules, and invoices), group the entire region into a single layout block with data-label="Table". Inside that block, use a proper <table> structure with <tr>, <th>, <td>, and correct colspan/rowspan to match merged cells.

{PROMPT_ENDING}
""".strip()

OCR_PROMPT = f"""
OCR this image to HTML.

{PROMPT_ENDING}
""".strip()

TABLE_ONLY_PROMPT = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.
You are given an image that contains only a table (or a single table region already cropped).
Return only the HTML for that table, nothing else. Do not wrap in extra <div> or layout markers.

Requirements:
- Output a single <table> with <thead>/<tbody>/<tr>/<th>/<td>. Add <caption> if present.
- Derive the canonical column count from the header and keep every row on those column boundaries.
- Preserve merged cells with exact colspan/rowspan. Do NOT duplicate text into hidden cells; cover them via spans.
- If stacked segments share aligned columns, merge them into one logical <table>.
- Keep empty cells as empty <td></td>. Never drop a cell.
- Keep numeric columns in their correct numeric columns; do not reorder. Use the header’s vertical guides to keep alignment.
- Preserve signs, separators, and units exactly (e.g., parentheses for negatives, thousand separators, %, ₫).
- Do not add boilerplate text or headings that are not visible.
- Reading order is top-to-bottom, left-to-right; preserve list markers/indentation as seen.
- Keep the logical structure of the table as shown in the image, no additional or editing is allowed.

{PROMPT_ENDING}
""".strip()
# TABLE_ONLY_PROMPT = f"""
# You are given an image that contains only a table (or a single table region already cropped).
# Return only the HTML for that table, nothing else. Do not wrap in extra <div> or layout markers

# {PROMPT_ENDING}
# """.strip()

PROMPT_MAPPING = {
    "ocr_layout": OCR_LAYOUT_PROMPT,
    "ocr": OCR_PROMPT,
    "ocr_table": TABLE_ONLY_PROMPT,
}


# * Tables (with or without visible borders):
#   - Always represent tabular data using <table>, <thead>, <tbody>, <tr>, <th>, <td>, and optionally <caption>.
#   - For tables with visible grid lines (borders), treat each cell separated by lines as a distinct <td> or <th>.
#   - For tables without visible borders, still detect columns using vertical alignment of text and numbers, and encode them as a table (not plain paragraphs).
#   - Determine the base number of columns from the header row (for example, “Cuối năm”, “Đầu năm”, …) and keep the structure consistent across rows.
#   - *Merged cells (very important):*
#     - If a cell visually spans multiple columns horizontally, use colspan to match the exact number of columns it covers.
#     - If a cell visually spans multiple rows vertically, use rowspan to match the exact number of rows it covers.
#     - Do NOT duplicate the same text into hidden cells. Represent it once with the appropriate colspan/rowspan.
#     - Even when a visual cell is merged, all underlying logical cells must still exist in the HTML structure (other rows must have the correct number of <td>, or use colspan/rowspan accordingly).
#   - Empty cells:
#     - If a cell appears empty in the image, still create an empty <td></td>. Do NOT drop or skip these cells.
#   - Financial tables, balance sheets, invoices:
#     - Lines like section headers (“1. Tiền”, “2. Các khoản đầu tư…”) that stretch across the entire row should be represented as a row with one <td> or <th> using colspan equal to the total number of columns.
#     - Sub-items (e.g. “- Tiền mặt”, “- Tiền gửi Ngân hàng không kỳ hạn”) should stay in the first column, with their corresponding numeric values in the correct numeric columns.
#   - Adjacent table segments:
#     - If multiple table regions are vertically stacked with aligned columns and only a small gap between them (i.e., they are visually a continuation of the same table), treat them as a single logical <table> and merge them into one table instead of separate tables.
#   - Avoid splitting one logical table into many smaller tables unless there is a clear visual separation.



# 
# - If the header repeats in the crop, include it once at the top.
