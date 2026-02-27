import concurrent.futures as cf
import json
import os
import random
import re
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import yaml
from tqdm import tqdm

try:
    import cv2
    import fitz
    import numpy as np

    HAS_RENDER_DEPS = True
except ImportError:
    cv2 = None
    fitz = None
    np = None
    HAS_RENDER_DEPS = False

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.utils import get_args


prompts = [
    "Please extract the document into a structured JSON key-value format.",
    "Convert the document content into JSON key-value pairs.",
    "Return the document as a JSON object with clear keys and values.",
    "Please provide the parsed document content in JSON format.",
]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def safe_int(value, default=1) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.current_cell = None
        self.current_row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "table":
            self.in_table = True
            return

        if not self.in_table:
            return

        if tag == "tr":
            self.in_row = True
            self.current_row = []
            return

        if tag in {"td", "th"} and self.in_row:
            self.current_cell = {
                "parts": [],
                "colspan": safe_int(attrs.get("colspan", 1)),
                "rowspan": safe_int(attrs.get("rowspan", 1)),
            }
            return

        if tag == "br" and self.current_cell is not None:
            self.current_cell["parts"].append("\n")

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            return

        if not self.in_table:
            return

        if tag in {"td", "th"}:
            self._finalize_cell()
            return

        if tag == "tr" and self.in_row:
            self._finalize_cell()
            self.rows.append(self.current_row)
            self.current_row = []
            self.in_row = False

    def handle_data(self, data):
        if self.current_cell is not None:
            self.current_cell["parts"].append(data)

    def _finalize_cell(self):
        if self.current_cell is None:
            return

        text = unescape("".join(self.current_cell["parts"]))
        text = normalize_whitespace(text).replace("|", "\\|")
        self.current_row.append(
            (
                text,
                self.current_cell["colspan"],
                self.current_cell["rowspan"],
            )
        )
        self.current_cell = None


def html_table_to_matrix(html: str) -> list[list[str]]:
    parser = TableParser()
    parser.feed(html)
    rows = parser.rows

    if not rows:
        return []

    matrix = []
    max_cols = 0

    for row in rows:
        col_count = sum(colspan for _, colspan, _ in row)
        max_cols = max(max_cols, col_count)

    for row_idx, row in enumerate(rows):
        if row_idx >= len(matrix):
            matrix.append([None] * max_cols)

        col_idx = 0

        for cell_text, colspan, rowspan in row:
            while col_idx < max_cols and matrix[row_idx][col_idx] is not None:
                col_idx += 1

            if col_idx >= max_cols:
                break

            for r in range(row_idx, min(row_idx + rowspan, len(rows))):
                while len(matrix) <= r:
                    matrix.append([None] * max_cols)

                for c in range(col_idx, min(col_idx + colspan, max_cols)):
                    # Duplicate merged-cell value across covered cells to preserve layout semantics.
                    matrix[r][c] = cell_text

            col_idx += colspan

    for row in matrix:
        while len(row) < max_cols:
            row.append("")
        for i in range(len(row)):
            if row[i] is None:
                row[i] = ""

    return matrix


def make_unique_headers(headers: list[str]) -> list[str]:
    seen = {}
    unique = []

    for idx, header in enumerate(headers):
        base = normalize_whitespace(header) or f"col_{idx + 1}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique_name = base if count == 1 else f"{base}_{count}"
        unique.append(unique_name)

    return unique


def html_table_to_kv(html: str) -> dict:
    matrix = html_table_to_matrix(html)
    if not matrix:
        return {"columns": [], "rows": []}

    headers = make_unique_headers(matrix[0])
    rows = []

    for row in matrix[1:]:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))

        row_obj = {
            headers[col_idx]: normalize_whitespace(row[col_idx])
            for col_idx in range(len(headers))
        }
        if any(value for value in row_obj.values()):
            rows.append(row_obj)

    return {"columns": headers, "rows": rows}


def formula_to_text(text: str) -> str:
    formula = text.replace("<div>", "\n").replace("</div>", "\n")
    formula = formula.replace("<span>", "\n").replace("</span>", "\n")
    formula = remove_html_comments(formula)
    return normalize_whitespace(formula)


IMAGE_LIKE_CATEGORIES = {
    "figure",
    "image",
    "logo",
    "stamp",
    "signature",
    "form",
    "header",
    "footer",
    "page_footnote",
}

KV_SEP_PATTERN = re.compile(r"^\s*([^:：]{1,180}?)\s*[:：]\s*(.+?)\s*$")
KV_MULTI_SPACE_PATTERN = re.compile(r"^\s*(.{1,180}?)\s{2,}(.+?)\s*$")
KV_MARKER_PATTERN = re.compile(r"([^:：]{1,180}?)\s*[:：]")
CHECKBOX_PATTERN = re.compile(r"([☑☐])\s*([^☑☐]+?)(?=(?:[☑☐])|$)")
EMPTY_KEY_PATTERN = re.compile(r"^\s*([^:：]{1,180}?)\s*[:：]\s*$")
KV_CHECKBOX_INLINE_PATTERN = re.compile(r"^\s*(.{1,180}?)\s+([☑☐].+)$")
TOTAL_LINE_PATTERN = re.compile(
    r"^\s*((?:tổng(?:\s*cộng)?|total(?:\s*(?:amount|sum))?|sum))\s*[:：\-]?\s*(.+?)\s*$",
    re.IGNORECASE,
)
SECTION_HINT_PATTERN = re.compile(
    r"(dành cho ngân hàng|for bank user only|for bank|người lĩnh tiền|receiver|đề nghị ghi nợ tài khoản|thông tin chi tiết|số tiền\s*\(with amount\))",
    re.IGNORECASE,
)
FORM_GRID_BLOCK_HINT_PATTERN = re.compile(
    r"(ký tên|signature|approval|duyệt|kiểm soát|for bank|dành cho ngân hàng|checkbox|tick)",
    re.IGNORECASE,
)
HEADER_HINT_PATTERN = re.compile(
    r"\b(date|ngày|time|reference|ref|stt|no\.?|debit|credit|balance|description|nội dung|amount|số tiền|qty|quantity|đơn giá|thành tiền|account|bank|counterparty|mô tả)\b",
    re.IGNORECASE,
)


def normalize_key(key: str) -> str:
    key = normalize_whitespace(key)
    key = re.sub(r"\s+", " ", key)
    return key.strip(":-–— \t")


def parse_checkbox_value(value: str) -> dict | None:
    matches = CHECKBOX_PATTERN.findall(value or "")
    if len(matches) < 2:
        return None
    checkbox = {}
    for checked, option in matches:
        key = normalize_key(option)
        if key:
            checkbox[key] = checked == "☑"
    return checkbox or None


def split_section_hint_from_key(key: str) -> tuple[str | None, str]:
    key = normalize_key(key)
    # Example: "Đề nghị ghi nợ tài khoản (...) Ngày (date)".
    dual_paren = re.match(r"(.+\([^)]+\))\s+([^:：]{1,50}\([^)]+\))$", key)
    if dual_paren:
        section_hint = normalize_key(dual_paren.group(1))
        leaf_key = normalize_key(dual_paren.group(2))
        if section_hint and leaf_key:
            return section_hint, leaf_key

    if len(key) < 60:
        return None, key

    suffix_with_paren = re.search(r"([^:：]{3,70}\([^)]+\))$", key)
    if suffix_with_paren and suffix_with_paren.start() > 8:
        section_hint = normalize_key(key[: suffix_with_paren.start()])
        leaf_key = normalize_key(suffix_with_paren.group(1))
        if section_hint and leaf_key:
            return section_hint, leaf_key
    return None, key


def parse_key_value_pairs(line: str) -> tuple[str | None, list[tuple[str, str | dict]]]:
    line = normalize_whitespace(line)
    if not line:
        return None, []

    matches = list(KV_MARKER_PATTERN.finditer(line))
    if not matches:
        match = KV_MULTI_SPACE_PATTERN.match(line)
        if not match:
            return None, []
        key = normalize_key(match.group(1))
        raw_value = normalize_whitespace(match.group(2))
        if not key or not raw_value:
            return None, []
        parsed_checkbox = parse_checkbox_value(raw_value)
        return None, [(key, parsed_checkbox if parsed_checkbox is not None else raw_value)]

    section_hint = None
    pairs = []

    for idx, match in enumerate(matches):
        raw_key = normalize_key(match.group(1))
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        raw_value = normalize_whitespace(line[value_start:value_end].strip(" -;|"))
        if not raw_key or not raw_value:
            continue

        inferred_section, leaf_key = split_section_hint_from_key(raw_key)
        if inferred_section:
            section_hint = inferred_section
            raw_key = leaf_key

        parsed_checkbox = parse_checkbox_value(raw_value)
        pairs.append((raw_key, parsed_checkbox if parsed_checkbox is not None else raw_value))

    if pairs:
        return section_hint, pairs

    # Fallback to single key-value parse.
    match = KV_SEP_PATTERN.match(line)
    if match:
        raw_key = normalize_key(match.group(1))
        raw_value = normalize_whitespace(match.group(2))
        if raw_key and raw_value:
            parsed_checkbox = parse_checkbox_value(raw_value)
            return None, [(raw_key, parsed_checkbox if parsed_checkbox is not None else raw_value)]

    # Fallback: inline checkbox line without ":".
    match = KV_CHECKBOX_INLINE_PATTERN.match(line)
    if match:
        raw_key = normalize_key(match.group(1))
        checkbox_obj = parse_checkbox_value(match.group(2))
        if raw_key and checkbox_obj:
            return None, [(raw_key, checkbox_obj)]

    return None, []


def looks_like_section_header(line: str) -> bool:
    line = normalize_whitespace(line)
    if not line:
        return False
    if ":" in line or "：" in line:
        return False
    if len(line) < 8:
        return False
    if len(line) > 90:
        return False
    if SECTION_HINT_PATTERN.search(line):
        return True
    alpha_chars = [ch for ch in line if ch.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars)
    if upper_ratio < 0.8 or len(alpha_chars) < 10:
        return False
    # Avoid numeric-only or mostly numeric rows.
    alpha_count = sum(ch.isalpha() for ch in line)
    digit_count = sum(ch.isdigit() for ch in line)
    if alpha_count == 0:
        return False
    if digit_count > alpha_count:
        return False
    return True


def looks_like_table_title(line: str) -> bool:
    line = normalize_whitespace(line)
    if not line:
        return False
    if ":" in line or "：" in line:
        return False
    if len(line) < 5 or len(line) > 120:
        return False
    if re.search(r"\b(table|bảng|danh sách|statement|chi tiết)\b", line, re.IGNORECASE):
        return True
    # Upper-ish short headings are often captions/titles.
    alpha_chars = [ch for ch in line if ch.isalpha()]
    if alpha_chars:
        upper_ratio = sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars)
        if upper_ratio > 0.75 and len(alpha_chars) >= 6:
            return True
    return False


def parse_empty_key_marker(line: str) -> str | None:
    match = EMPTY_KEY_PATTERN.match(normalize_whitespace(line))
    if not match:
        return None
    key = normalize_key(match.group(1))
    return key or None


def is_total_like_key(key: str) -> bool:
    key = normalize_key(key).lower()
    if not key:
        return False
    return any(token in key for token in ["tổng", "total", "sum"])


def parse_total_line(line: str) -> tuple[str, str] | None:
    line = normalize_whitespace(line)
    if not line:
        return None
    match = TOTAL_LINE_PATTERN.match(line)
    if not match:
        return None

    total_key = normalize_key(match.group(1))
    total_value = normalize_whitespace(match.group(2))
    if not total_key or not total_value:
        return None
    return total_key, total_value


def cell_is_value_like(cell: str) -> bool:
    text = normalize_whitespace(cell)
    if not text:
        return False
    if "☑" in text or "☐" in text:
        return True
    alpha_count = sum(ch.isalpha() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        return True
    if digit_count > 0:
        if alpha_count == 0:
            return True
        if digit_count >= 3:
            return True
        if digit_count > alpha_count:
            return True
    if re.search(r"[\$€£¥₫%]", text):
        return True
    if len(text) > 45:
        return True
    if re.match(r"^[\d,./\-()]+$", text):
        return True
    return False


def cell_looks_like_header(cell: str) -> bool:
    text = normalize_whitespace(cell)
    if not text:
        return False
    if HEADER_HINT_PATTERN.search(text):
        return True
    if cell_is_value_like(text):
        return False
    if len(text.split()) <= 6 and len(text) <= 40:
        return True
    return False


def is_structured_table_matrix(matrix: list[list[str]]) -> bool:
    if not matrix:
        return False

    rows = [[normalize_whitespace(cell) for cell in row] for row in matrix]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return False

    active_col_indices = [idx for idx in range(len(rows[0])) if any(row[idx] for row in rows)]
    if len(active_col_indices) < 2:
        return False
    rows = [[row[idx] for idx in active_col_indices] for row in rows]

    col_count = len(rows[0])
    if col_count < 2:
        return False

    all_cells = [cell for row in rows for cell in row if cell]
    if not all_cells:
        return False

    checkbox_cells = sum(1 for cell in all_cells if "☑" in cell or "☐" in cell)
    if checkbox_cells >= 2:
        return False

    if any(FORM_GRID_BLOCK_HINT_PATTERN.search(cell) for cell in all_cells):
        return False

    header = rows[0]
    header_non_empty = [cell for cell in header if cell]
    if len(header_non_empty) < 2:
        return False

    header_kv_ratio = sum(1 for cell in header_non_empty if ":" in cell or "：" in cell) / len(header_non_empty)
    if header_kv_ratio > 0.4:
        return False

    header_value_like_ratio = sum(1 for cell in header_non_empty if cell_is_value_like(cell)) / len(header_non_empty)
    if header_value_like_ratio > 0.4:
        return False

    body = rows[1:]
    if not body:
        return False

    body_non_empty_counts = [sum(1 for cell in row if cell) for row in body]
    valid_record_rows = [cnt for cnt in body_non_empty_counts if cnt >= 2]
    if len(valid_record_rows) < 2:
        return False

    if valid_record_rows:
        min_cnt = min(valid_record_rows)
        max_cnt = max(valid_record_rows)
        if min_cnt == 0 or (max_cnt / max(min_cnt, 1)) > 4:
            return False

    body_cells = [cell for row in body for cell in row if cell]
    if body_cells:
        kv_ratio = sum(1 for cell in body_cells if ":" in cell or "：" in cell) / len(body_cells)
        if kv_ratio > 0.5:
            return False

    # Two-column matrices are often form grids; require a clearer header signal.
    if col_count == 2:
        header_hint_count = sum(1 for cell in header_non_empty if cell_looks_like_header(cell))
        if header_hint_count < 1:
            return False

    return True


def table_matrix_to_text_lines(matrix: list[list[str]]) -> list[str]:
    lines = []
    for row in matrix:
        for cell in row:
            clean_cell = normalize_whitespace(cell)
            if clean_cell:
                lines.append(clean_cell)
    return lines


def form_grid_matrix_to_lines(matrix: list[list[str]]) -> list[str]:
    lines = []
    for row in matrix:
        cells = [normalize_whitespace(cell) for cell in row if normalize_whitespace(cell)]
        if not cells:
            continue

        # Keep rows that are already complete key-value cells as-is.
        if all(parse_key_value_pairs(cell)[1] for cell in cells):
            lines.extend(cells)
            continue

        if len(cells) >= 2:
            key = cells[0]
            value = normalize_whitespace(" ".join(cells[1:]))
            if value:
                lines.append(f"{key}: {value}")
            else:
                lines.append(key)
        else:
            lines.append(cells[0])
    return lines


def extract_table_caption_and_matrix(matrix: list[list[str]]) -> tuple[str | None, list[list[str]]]:
    if not matrix:
        return None, matrix

    rows = [[normalize_whitespace(cell) for cell in row] for row in matrix]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return None, rows

    first_non_empty = [cell for cell in rows[0] if cell]
    second_non_empty_count = sum(1 for cell in rows[1] if cell)
    unique_first = set(first_non_empty)

    # Caption pattern: first row has a single text cell, next row looks like table header.
    if len(unique_first) == 1 and second_non_empty_count >= 2:
        caption = normalize_key(next(iter(unique_first)))
        return (caption if caption else None), rows[1:]

    return None, rows


def matrix_to_table_kv(matrix: list[list[str]]) -> dict:
    if not matrix:
        return {"columns": [], "rows": []}

    headers = make_unique_headers(matrix[0])
    rows = []

    for row in matrix[1:]:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))

        row_obj = {
            headers[col_idx]: normalize_whitespace(row[col_idx])
            for col_idx in range(len(headers))
        }
        if any(value for value in row_obj.values()):
            rows.append(row_obj)

    return {"columns": headers, "rows": rows}


def add_unique_key_value(label_obj: dict, key: str, value) -> str | None:
    key = normalize_key(key)
    if isinstance(value, str):
        value = normalize_whitespace(value)
    if not key:
        return None
    if isinstance(value, str) and not value:
        return None

    if key not in label_obj:
        label_obj[key] = value
        return key

    if label_obj[key] == value:
        return key

    suffix = 2
    while f"{key}_{suffix}" in label_obj:
        suffix += 1
    new_key = f"{key}_{suffix}"
    label_obj[new_key] = value
    return new_key


class LabelWriter:
    def __init__(self, nested: bool = False):
        self.nested = nested
        self.root = {}
        self.current_section = None
        self.free_text_counter = {"__root__": 1}
        self.table_counter = {"__root__": 1}
        self.pending_table_name = {}
        self.last_table_key_by_scope = {}
        self.allow_total_row_by_scope = {}

    def _section_scope_key(self, section: str | None) -> str:
        return section or "__root__"

    def close_total_window(self, section: str | None = None):
        if not self.nested:
            section = None
        scope = self._section_scope_key(section)
        if scope in self.allow_total_row_by_scope:
            self.allow_total_row_by_scope[scope] = False

    def _ensure_unique_key(self, container: dict, base_key: str) -> str:
        key = normalize_key(base_key) or "Field"
        if key not in container:
            return key
        suffix = 2
        while f"{key}_{suffix}" in container:
            suffix += 1
        return f"{key}_{suffix}"

    def _next_free_text_key(self, section: str | None) -> str:
        scope = self._section_scope_key(section)
        if scope not in self.free_text_counter:
            self.free_text_counter[scope] = 1
        idx = self.free_text_counter[scope]
        self.free_text_counter[scope] += 1
        return f"FreeText{idx}"

    def _next_table_key(self, section: str | None) -> str:
        scope = self._section_scope_key(section)
        if scope not in self.table_counter:
            self.table_counter[scope] = 1
        idx = self.table_counter[scope]
        self.table_counter[scope] += 1
        return f"Table{idx}"

    def _ensure_section(self, section_name: str) -> str:
        section_name = normalize_key(section_name)
        if not section_name:
            return ""

        final_name = section_name
        if final_name in self.root and not isinstance(self.root[final_name], dict):
            suffix = 2
            while f"{section_name}_{suffix}" in self.root:
                suffix += 1
            final_name = f"{section_name}_{suffix}"

        if final_name not in self.root:
            self.root[final_name] = {}
        return final_name

    def _container(self, section: str | None):
        if not self.nested or not section:
            return self.root
        return self.root.get(section, self.root)

    def set_title(self, text: str):
        text = normalize_whitespace(text)
        if not text:
            return
        if "Title" not in self.root:
            self.root["Title"] = text
            return
        self.add_free_text(text, section=self.current_section if self.nested else None)

    def set_current_section(self, section_name: str):
        if not self.nested:
            return
        section_key = self._ensure_section(section_name)
        if section_key:
            self.current_section = section_key
            self.close_total_window(section_key)

    def set_pending_table_name(self, title: str, section: str | None = None):
        title = normalize_key(title)
        if not title:
            return
        if not self.nested:
            section = None
        scope = self._section_scope_key(section)
        self.pending_table_name[scope] = title

    def add_free_text(self, text: str, section: str | None = None):
        text = normalize_whitespace(text)
        if not text:
            return
        if not self.nested:
            section = None
        self.close_total_window(section)
        key = self._next_free_text_key(section)
        container = self._container(section)
        container[key] = text

    def add_key_value(self, key: str, value, section: str | None = None):
        if not self.nested:
            section = None
        self.close_total_window(section)
        container = self._container(section)
        add_unique_key_value(container, key, value)

    def add_table(self, rows: list[dict], section: str | None = None, table_name: str | None = None):
        if not self.nested:
            section = None
        scope = self._section_scope_key(section)
        container = self._container(section)
        if table_name:
            key = self._ensure_unique_key(container, table_name)
        else:
            pending_name = self.pending_table_name.pop(scope, None)
            if pending_name:
                key = self._ensure_unique_key(container, pending_name)
            else:
                key = self._next_table_key(section)
        container[key] = rows
        self.last_table_key_by_scope[scope] = key
        self.allow_total_row_by_scope[scope] = True
        return key

    def append_total_row(self, total_key: str, total_value: str, section: str | None = None) -> bool:
        if not self.nested:
            section = None
        scope = self._section_scope_key(section)
        if not self.allow_total_row_by_scope.get(scope, False):
            return False
        table_key = self.last_table_key_by_scope.get(scope)
        if not table_key:
            return False

        container = self._container(section)
        table_rows = container.get(table_key)
        if not isinstance(table_rows, list) or not table_rows:
            return False
        if not isinstance(table_rows[0], dict):
            return False

        headers = list(table_rows[0].keys())
        if not headers:
            return False

        total_key_clean = normalize_key(total_key)
        total_value_clean = normalize_whitespace(total_value)
        if not total_key_clean:
            return False

        row = {h: "" for h in headers}

        total_header = None
        for h in headers:
            hl = h.lower()
            if "tổng" in hl or "total" in hl:
                total_header = h
                break
        if total_header is None:
            total_header = headers[0]
        row[total_header] = total_key_clean

        value_header = None
        for h in reversed(headers):
            hl = h.lower()
            if any(token in hl for token in ["amount", "thành tiền", "số tiền", "total", "balance", "credit", "debit"]):
                value_header = h
                break
        if value_header is None:
            value_header = headers[-1]
        if total_value_clean:
            row[value_header] = total_value_clean

        if table_rows and table_rows[-1] == row:
            return True

        table_rows.append(row)
        self.allow_total_row_by_scope[scope] = True
        return True


def put_text_lines_into_label(writer: LabelWriter, lines: list[str]):
    inserted_any = False
    orphan_lines = []
    pending_key = None

    for line in lines:
        clean_line = normalize_whitespace(line)
        if not clean_line:
            continue

        if pending_key:
            if writer.nested and normalize_key(clean_line) == normalize_key(pending_key):
                writer.set_current_section(pending_key)
                inserted_any = True
                pending_key = None
                continue

            section_hint_after_pending, kv_pairs_after_pending = parse_key_value_pairs(clean_line)
            if writer.nested and kv_pairs_after_pending:
                writer.set_current_section(pending_key)
                target_section = writer.current_section
                if section_hint_after_pending:
                    writer.set_current_section(section_hint_after_pending)
                    target_section = writer.current_section
                for key, value in kv_pairs_after_pending:
                    writer.add_key_value(key, value, section=target_section)
                inserted_any = True
                pending_key = None
                continue

            checkbox_obj = parse_checkbox_value(clean_line)
            if checkbox_obj:
                writer.add_key_value(
                    pending_key,
                    checkbox_obj,
                    section=writer.current_section if writer.nested else None,
                )
                inserted_any = True
                pending_key = None
                continue

            writer.add_key_value(
                pending_key,
                clean_line,
                section=writer.current_section if writer.nested else None,
            )
            inserted_any = True
            pending_key = None
            continue

        parsed_total = parse_total_line(clean_line)
        if parsed_total:
            total_key, total_value = parsed_total
            target_section = writer.current_section if writer.nested else None
            if writer.append_total_row(total_key, total_value, section=target_section):
                inserted_any = True
                continue

        if writer.nested and looks_like_section_header(clean_line):
            writer.close_total_window(section=writer.current_section if writer.nested else None)
            writer.set_current_section(clean_line)
            inserted_any = True
            continue

        if looks_like_table_title(clean_line):
            writer.close_total_window(section=writer.current_section if writer.nested else None)
            writer.set_pending_table_name(
                clean_line,
                section=writer.current_section if writer.nested else None,
            )
            inserted_any = True
            continue

        empty_key = parse_empty_key_marker(clean_line)
        if empty_key:
            writer.close_total_window(section=writer.current_section if writer.nested else None)
            pending_key = empty_key
            continue

        section_hint, kv_pairs = parse_key_value_pairs(clean_line)
        if kv_pairs:
            target_section = writer.current_section if writer.nested else None
            if writer.nested and section_hint:
                writer.set_current_section(section_hint)
                target_section = writer.current_section

            if len(kv_pairs) == 1:
                only_key, only_value = kv_pairs[0]
                if isinstance(only_value, str) and is_total_like_key(only_key):
                    if writer.append_total_row(only_key, only_value, section=target_section):
                        inserted_any = True
                        continue

            for key, value in kv_pairs:
                writer.add_key_value(key, value, section=target_section)
            inserted_any = True
        else:
            orphan_lines.append(clean_line)

    if orphan_lines:
        writer.add_free_text(
            "\n".join(orphan_lines),
            section=writer.current_section if writer.nested else None,
        )
    elif not inserted_any and lines and not pending_key:
        writer.add_free_text(
            "\n".join(lines),
            section=writer.current_section if writer.nested else None,
        )

    if pending_key:
        if writer.nested and (
            looks_like_section_header(pending_key) or SECTION_HINT_PATTERN.search(pending_key)
        ):
            writer.set_current_section(pending_key)
        elif writer.nested:
            writer.add_free_text(f"{pending_key}:", section=writer.current_section if writer.nested else None)
        else:
            writer.add_key_value(pending_key, "")


def collect_source_lines(form_items: list[dict]) -> list[str]:
    source_lines = []

    for item in form_items:
        category = str(item.get("category", "") or "")
        if category in IMAGE_LIKE_CATEGORIES:
            continue

        raw_text = str(item.get("text", "") or "")
        if not raw_text:
            continue

        if category == "table":
            matrix = html_table_to_matrix(raw_text)
            for row in matrix:
                for cell in row:
                    clean_cell = normalize_whitespace(cell)
                    if not clean_cell:
                        continue
                    for line in clean_cell.split("\n"):
                        clean_line = normalize_whitespace(line)
                        if clean_line:
                            source_lines.append(clean_line)
            continue

        if category == "formula":
            normalized = formula_to_text(raw_text)
        else:
            normalized = normalize_whitespace(raw_text)

        if not normalized:
            continue

        for line in normalized.split("\n"):
            clean_line = normalize_whitespace(line)
            if clean_line:
                source_lines.append(clean_line)

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for line in source_lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return deduped


def line_is_covered_by_label(line: str, label_blob: str) -> bool:
    if line in label_blob:
        return True

    checkbox_obj = parse_checkbox_value(line)
    if checkbox_obj:
        for option_key in checkbox_obj.keys():
            if normalize_key(option_key) not in label_blob:
                return False
        return True

    section_hint, kv_pairs = parse_key_value_pairs(line)
    if kv_pairs:
        for key, value in kv_pairs:
            if normalize_key(key) not in label_blob:
                return False
            if isinstance(value, dict):
                for option_key in value.keys():
                    if normalize_key(option_key) not in label_blob:
                        return False
            else:
                if normalize_whitespace(str(value)) not in label_blob:
                    return False
        return True

    maybe_empty_key = parse_empty_key_marker(line)
    if maybe_empty_key:
        return maybe_empty_key in label_blob

    return False


def preserve_missing_lines(writer: LabelWriter, form_items: list[dict]) -> None:
    source_lines = collect_source_lines(form_items)
    if not source_lines:
        return

    label_blob = normalize_whitespace(json.dumps(writer.root, ensure_ascii=False))
    missing_lines = []

    for line in source_lines:
        if not line_is_covered_by_label(line, label_blob):
            missing_lines.append(line)

    if missing_lines:
        # Keep uncaptured content in root-level free text so conversion is lossless.
        writer.add_free_text("\n".join(missing_lines), section=None)


def form_to_kv_label(form_items: list[dict], label_style: str = "nested") -> dict:
    nested = str(label_style or "").lower() == "nested"
    writer = LabelWriter(nested=nested)

    for item in form_items:
        category = str(item.get("category", "") or "")
        text = str(item.get("text", "") or "")

        if category in IMAGE_LIKE_CATEGORIES:
            continue

        if category == "title":
            writer.set_title(text)
            continue

        if category == "table":
            matrix = html_table_to_matrix(text)
            caption, matrix = extract_table_caption_and_matrix(matrix)
            if caption:
                writer.set_pending_table_name(
                    caption,
                    section=writer.current_section if nested else None,
                )

            if is_structured_table_matrix(matrix):
                table = matrix_to_table_kv(matrix)
                rows = table.get("rows", []) if isinstance(table, dict) else []
                columns = table.get("columns", []) if isinstance(table, dict) else []
                if rows:
                    writer.add_table(
                        rows,
                        section=writer.current_section if nested else None,
                        table_name=caption,
                    )
                elif columns:
                    writer.add_table(
                        [],
                        section=writer.current_section if nested else None,
                        table_name=caption,
                    )
            else:
                lines = form_grid_matrix_to_lines(matrix)
                if not lines:
                    lines = table_matrix_to_text_lines(matrix)
                put_text_lines_into_label(writer, lines)
            continue

        if category == "formula":
            writer.add_free_text(
                formula_to_text(text),
                section=writer.current_section if nested else None,
            )
            continue

        lines = [normalize_whitespace(line) for line in normalize_whitespace(text).split("\n")]
        put_text_lines_into_label(writer, [line for line in lines if line])

    preserve_missing_lines(writer, form_items)
    return writer.root


def has_meaningful_label(label_obj: dict) -> bool:
    if not isinstance(label_obj, dict) or not label_obj:
        return False

    for value in label_obj.values():
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and len(value) > 0:
            return True
        if isinstance(value, dict) and len(value) > 0:
            return True

    return False


def render_page_job(item, dpi: int = 180) -> str | None:
    if not HAS_RENDER_DEPS:
        return None

    try:
        img_path = Path(item["image"])
        pdf_path = str(img_path) + ".pdf"

        with fitz.open(pdf_path) as doc:
            pix = doc.load_page(0).get_pixmap(dpi=dpi)
            if not img_path.exists():
                pix.save(img_path)
            if has_red_cv(img_path):
                return None
            return str(img_path)
    except Exception as exc:
        print(exc)
        return None


def has_red_cv(image_path, ratio=0.001) -> bool:
    if cv2 is None or np is None:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    red_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
    return red_ratio > ratio


def render_jobs_multiprocess(jobs, dpi=180, max_workers=40) -> list[str]:
    if not HAS_RENDER_DEPS:
        print("Rendering skipped: missing optional deps (fitz/cv2/numpy).")
        return []

    with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(render_page_job, job, dpi=dpi) for job in jobs]
        results = []
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Rendering"):
            result = fut.result()
            if result:
                results.append(result)
    return results


def merge_json_files(input_path, output_path) -> None:
    input_dir = os.path.dirname(input_path)
    output_name = os.path.basename(output_path)
    merged_data = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json") or filename == output_name:
            continue

        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(merged_data, file, indent=2, ensure_ascii=False)


def form2docparse(datas: list[dict], label_style: str = "nested") -> tuple[list[dict], list[dict]]:
    weights = [8] + [1.0] * (len(prompts) - 1)
    results = []
    labels_per_image = []
    skipped_empty = 0

    for data in tqdm(datas):
        image = data["image"]
        form_items = data.get("form", [])
        label_obj = form_to_kv_label(form_items, label_style=label_style)
        if not has_meaningful_label(label_obj):
            skipped_empty += 1
            continue
        label_json = json.dumps(label_obj, ensure_ascii=False, indent=2)
        labels_per_image.append({"image": image, "label": label_obj})

        results.append(
            {
                "images": [image],
                "conversations": [
                    {
                        "from": "human",
                        "value": random.choices(prompts, weights=weights, k=1)[0],
                    },
                    {
                        "from": "gpt",
                        "value": label_json,
                    },
                ],
            }
        )

    if skipped_empty:
        print(f"Skipped {skipped_empty} empty labels.")

    return results, labels_per_image


def save_labels_per_image(labels_per_image: list[dict], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(labels_per_image):
        image_path = str(item.get("image", "") or "")
        stem = Path(image_path).stem if image_path else f"id{idx}"
        label = item.get("label", {})
        save_path = output_path / f"{stem}.json"

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(label, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(labels_per_image)} label files to: {output_path}")


if __name__ == "__main__":
    args = get_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    gt_path = config["work_path"]["output_gt_path"]
    temp_path = os.path.join(os.path.dirname(gt_path), "temp.json")
    merge_json_files(gt_path, temp_path)

    with open(temp_path, "r", encoding="utf-8") as file:
        form_data = json.load(file)

    render_jobs_multiprocess(form_data, max_workers=40)
    label_style = config.get("work_path", {}).get("label_style", "nested")
    result, labels_per_image = form2docparse(form_data, label_style=label_style)

    with open(config["work_path"]["result"], "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    label_output_dir = config["work_path"].get("label_output_dir")
    if not label_output_dir:
        result_parent = Path(config["work_path"]["result"]).parent
        label_output_dir = str(result_parent / "labels")
    save_labels_per_image(labels_per_image, label_output_dir)
