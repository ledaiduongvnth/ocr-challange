import os
import random
from pprint import pformat


def _pick_first_block(value):
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, dict):
        return value
    return None


def _rect(block):
    try:
        return (
            float(block.get("x", 0)),
            float(block.get("y", 0)),
            float(block.get("w", 0)),
            float(block.get("h", 0)),
        )
    except Exception:
        return None


def _layout_blocks(blocks, page_width, top, bottom, left, right, row_gap, col_gap):
    x = left
    y = top
    row_h = 0.0
    for b in blocks:
        w = float(b.get("w", 0) or 0)
        h = float(b.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            continue
        if x + w > page_width - right:
            x = left
            y += row_h + row_gap
            row_h = 0.0
        if y + h > bottom:
            raise ValueError("Layout overflow: reduce element counts or sizes to fit page.")
        b["x"] = x
        b["y"] = y
        x += w + col_gap
        if h > row_h:
            row_h = h


def _block_key(b):
    return (
        b.get("type"),
        b.get("header"),
        b.get("header_boxed"),
        b.get("content"),
        b.get("src"),
        b.get("w"),
        b.get("h"),
    )


def get_data(self, layout_config):
    elem_cfg = layout_config.get("element", {})
    blocks = []

    mapping = {
        "logo": "logo_iter",
        "form": "form_iter",
        "text": "text_iter",
        "table": "table_iter",
        "stamp": "stamp_iter",
        "signature": "signature_iter",
        "figure": "figure_iter",
        "image": "figure_iter",
        "title": "title_iter",
        "formula": "formula_iter",
    }

    randomize_counts = bool(layout_config.get("randomize_counts", False))
    shuffle_blocks = layout_config.get("shuffle", True)

    seen = set()
    for key, iter_name in mapping.items():
        max_count = elem_cfg.get(key, 0)
        if key == "table" and max_count == 0:
            max_count = 1
        if not max_count:
            continue
        it = getattr(self, iter_name, None)
        if it is None:
            continue
        if key == "table" and not self.table:
            raise ValueError("Table data missing: provide data_paths.table with at least one item.")
        insert_count = random.randint(0, max_count) if randomize_counts else max_count
        added = 0
        attempts = 0
        max_attempts = max(10, insert_count * 5)
        while added < insert_count and attempts < max_attempts:
            attempts += 1
            candidate = next(it)
            k = _block_key(candidate)
            if k in seen:
                continue
            seen.add(k)
            blocks.append(candidate)
            added += 1

    if shuffle_blocks:
        random.shuffle(blocks)

    header_block = next(self.header_iter, None) if self.header else None
    if header_block is None:
        header_block = _pick_first_block(layout_config.get("header"))
    if header_block is None:
        raise ValueError("Header data missing: provide data_paths.header or layout_config.header")

    footer_block = next(self.footer_iter, None) if self.footer else None
    if footer_block is None:
        footer_block = _pick_first_block(layout_config.get("footer"))
    if footer_block is None:
        raise ValueError("Footer data missing: provide data_paths.footer or layout_config.footer")

    header_rect = _rect(header_block)
    footer_rect = _rect(footer_block)

    page_w = float(layout_config.get("page_width", 2433))
    left = float(layout_config.get("left_margin", 120))
    right = float(layout_config.get("right_margin", 120))
    row_gap = float(layout_config.get("row_gap", 16))
    col_gap = float(layout_config.get("col_gap", 16))
    margin = float(layout_config.get("header_footer_margin", 8))

    top = (header_rect[1] + header_rect[3] + margin) if header_rect else margin
    bottom = (footer_rect[1] - margin) if footer_rect else float(layout_config.get("page_height", 3508)) - margin

    _layout_blocks(blocks, page_w, top, bottom, left, right, row_gap, col_gap)
    blocks = [header_block] + blocks + [footer_block]

    pretty = pformat(blocks, width=120)
    print(pretty)
    debug_path = layout_config.get("debug_blocks_path")
    if debug_path:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(pretty + "\n")
    return {"blocks": blocks}
