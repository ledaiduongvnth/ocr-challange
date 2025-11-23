# ALLOWED_TAGS = [
#     "math",
#     "br",
#     "i",
#     "b",
#     "u",
#     "del",
#     "sup",
#     "sub",
#     "table",
#     "tr",
#     "td",
#     "p",
#     "th",
#     "div",
#     "pre",
#     "h1",
#     "h2",
#     "h3",
#     "h4",
#     "h5",
#     "ul",
#     "ol",
#     "li",
#     "input",
#     "a",
#     "span",
#     "img",
#     "hr",
#     "tbody",
#     "small",
#     "caption",
#     "strong",
#     "thead",
#     "big",
#     "code",
# ]
# ALLOWED_ATTRIBUTES = [
#     "class",
#     "colspan",
#     "rowspan",
#     "display",
#     "checked",
#     "type",
#     "border",
#     "value",
#     "style",
#     "href",
#     "alt",
#     "align",
# ]

# PROMPT_ENDING = f"""
# Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

# Guidelines:
# * Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
# * Tables: Use colspan and rowspan attributes to match table structure. If a cell in the table is empty (in terms of content, specifically no text or empty string). Make sure that cell still exists in the markdown. Beware of tables without borders
# * Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
# * Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
# * Forms: Mark checkboxes and radio buttons properly.
# * Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
# * Use the simplest possible HTML structure that accurately represents the content of the block.
# * Make sure the text is accurate and easy for a human to read and interpret. Reading order should be correct and natural.
# * In header or footer content, lines can easily get stuck together. Remember to separate them. You often make mistakes there.
# """.strip()

# OCR_LAYOUT_PROMPT = f"""
# OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format.  Bboxes are normalized 0-{{bbox_scale}}. The data-label attribute is the label for the block.

# Use the following labels:
# - Caption
# - Footnote
# - Equation-Block
# - List-Group
# - Page-Header
# - Page-Footer
# - Image
# - Section-Header
# - Table
# - Text
# - Complex-Block
# - Code-Block
# - Form
# - Table-Of-Contents
# - Figure

# {PROMPT_ENDING}
# """.strip()

# OCR_PROMPT = f"""
# OCR this image to HTML.

# {PROMPT_ENDING}
# """.strip()

# PROMPT_MAPPING = {
#     "ocr_layout": OCR_LAYOUT_PROMPT,
#     "ocr": OCR_PROMPT,
# }


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
* Tables (including tables without visible borders):
  - Always represent tables using <table>, <tr>, <th>, <td>, <tbody>, <thead>, <caption> tags.
  - For tables without borders (cells only separated by spacing/alignment), still treat them as tables, not as plain text.
  - Determine columns by visual alignment: content that is vertically aligned should belong to the same column.
  - Ensure each row has a consistent number of columns unless there is a clear visual reason to use colspan/rowspan.
  - Use colspan and rowspan attributes to match table structure when cells are visually merged.
  - If a cell is visually empty (no text), still create a <td></td> for it. Do NOT skip empty cells.
  - Do not merge multiple logically separate cells into one just because they are close together; keep numeric columns and labels in separate <td> when they are in separate visual columns.
  - Beware of tables without borders: invoices, financial reports, or forms often have column alignment but no drawn grid. These should still be encoded as proper HTML tables.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
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

For tables (including those without visible borders), group the entire table region into a single layout block with data-label="Table". Inside that block, use proper <table>, <tr>, <th>, and <td> tags to represent the grid structure, even if no borders are drawn in the image.

{PROMPT_ENDING}
""".strip()

OCR_PROMPT = f"""
OCR this image to HTML.

{PROMPT_ENDING}
""".strip()

PROMPT_MAPPING = {
    "ocr_layout": OCR_LAYOUT_PROMPT,
    "ocr": OCR_PROMPT,
}
