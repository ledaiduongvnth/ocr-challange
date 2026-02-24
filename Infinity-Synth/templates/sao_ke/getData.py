import os
import random
from pprint import pformat


def _pick_first_block(value):
    # Lấy phần tử đầu tiên nếu là list, hoặc trả về dict nếu đã là dict.
    # Dùng để fallback header/footer khi không có iterator.
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, dict):
        return value
    return None


def _rect(block):
    # Trích (x, y, w, h) từ block để tính vùng header/footer.
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
    # Fixed layout theo mẫu A4
    content_w = page_width - left - right
    content_h = bottom - top

    # ===== Height ratios =====
    row1_h = content_h * 0.06
    row2_h = content_h * 0.05
    form_h = content_h * 0.20
    # Giữ title trước table ở chiều cao cố định, để table nằm sát ngay bên dưới.
    title1_h = 40.0
    table_h = content_h * 0.20
    title2_h = content_h * 0.015
    checklist_h = content_h * 0.16
    title3_h = content_h * 0.015
    textbar_h = content_h * 0.04

    used = (
        row1_h + row2_h + form_h +
        title1_h + table_h +
        title2_h + checklist_h +
        title3_h + textbar_h
    )
    bottom_h = max(0.0, content_h - used)

    # ===== X layout =====
    # Thu nhỏ vùng logo để khớp kích thước logo thực tế hơn
    logo_w = content_w * 0.28
    right_w = content_w - logo_w
    title_w = right_w / 3.0

    # ===== Y layout =====
    y0 = top
    y1 = y0 + row1_h
    y2 = y1 + row2_h
    y3 = y2 + form_h
    y4 = y3 + title1_h
    y5 = y4 + table_h
    y6 = y5 + title2_h
    y7 = y6 + checklist_h
    y8 = y7 + title3_h
    y9 = y8 + textbar_h

    # ===== Slots =====
    logo_h = row1_h * 0.85
    logo_y = y0 + (row1_h - logo_h) / 2.0

    slots = {
        "logo": [(left, logo_y, logo_w, logo_h)],
        "title": [
            # Dời 3 title ngang xuống hàng dưới (vị trí header cũ)
            (left + logo_w + 0 * title_w, y1, title_w, row2_h),
            (left + logo_w + 1 * title_w, y1, title_w, row2_h),
            (left + logo_w + 2 * title_w, y1, title_w, row2_h),
            (left, y3, content_w, title1_h),
            (left, y5, content_w, title2_h),
            (left, y7, content_w, title3_h),
        ],
        "text": [
            # Bỏ header cũ ở hàng y1 (không dùng slot text tại đây nữa)
            (left, y8, content_w, textbar_h),
            (left, y9, content_w * 0.60, bottom_h),
        ],
        "form": [(left, y2, content_w, form_h)],
        "table": [(left, y4, content_w, table_h)],
        "checklist": [(left, y6, content_w, checklist_h)],
        "signature": [(left + content_w * 0.60, y9, content_w * 0.40, bottom_h)],
        "stamp": [],
        "image": [],
        "figure": [],
        "formula": [],
    }

    slot_idx = {k: 0 for k in slots}

    # ===== Assign coordinates =====
    # Chỉ giữ các block có slot hợp lệ để tránh block dư dùng tọa độ gốc JSON.
    placed_blocks = []
    for b in blocks:
        b_type = b.get("type")
        # Watermark giữ nguyên tọa độ gốc từ data JSON, không ép theo slot.
        if b_type in {"watermark", "watermark_image"} or b.get("role") == "watermark":
            placed_blocks.append(b)
            continue
        if b_type not in slots or not slots[b_type]:
            continue

        idx = slot_idx[b_type]
        if idx >= len(slots[b_type]):
            continue

        x, y, w, h = slots[b_type][idx]
        b["x"] = x
        b["y"] = y
        b["w"] = w
        b["h"] = h
        slot_idx[b_type] += 1
        placed_blocks.append(b)

    # Separator chỉ thêm khi bật explicit trong config
    add_separators = bool(getattr(_layout_blocks, "_add_separators", False))
    if add_separators:
        sep_h = 24
        sections = [
            ("form", y3),
            ("table", y5),
            ("checklist", y7),
        ]

        for name, y_sep in sections:
            sep_html = f"""
            <div style="
                width:100%;
                text-align:center;
                font-family:monospace;
                font-size:12px;
                color:#444;
                line-height:{sep_h}px;
            ">
                ----------- {name} -----------
            </div>
            """

            placed_blocks.append({
                "type": "text",
                "x": left,
                "y": y_sep - sep_h / 2,
                "w": content_w,
                "h": sep_h,
                "content": sep_html,
                "header": None,
                "header_boxed": False,
            })

    return placed_blocks

def _block_key(b):
    # Tạo khóa để tránh add trùng block vào layout.
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
    # Entry chính: tạo danh sách blocks theo layout_config và data input.
    elem_cfg = layout_config.get("element", {})
    blocks = []

    mapping = {
        "logo": "logo_iter",
        "form": "form_iter",
        "text": "text_iter",
        "table": "table_iter",
        "checklist": "checklist_iter",
        "stamp": "stamp_iter",
        "signature": "signature_iter",
        "figure": "figure_iter",
        "image": "figure_iter",
        "title": "title_iter",
        "formula": "formula_iter",
    }

    # Nếu bật randomize_counts thì số lượng mỗi loại block sẽ random trong [0..max]
    randomize_counts = bool(layout_config.get("randomize_counts", False))
    # Có thể trộn thứ tự block trước khi layout
    shuffle_blocks = layout_config.get("shuffle", True)

    # Duyệt từng loại block theo mapping và thêm vào danh sách blocks
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
            try:
                candidate = next(it)
            except StopIteration:
                # Iterator có thể là finite; dừng mềm thay vì làm crash pipeline.
                break
            # Chuẩn hóa type theo slot key để không bị lệch layout khi data json để type="block".
            candidate = dict(candidate)
            # Riêng stamp có thể dùng như watermark toàn trang; giữ nguyên bằng role watermark.
            if key == "stamp" and ("opacity" in str(candidate.get("style", "")) or "filter" in str(candidate.get("style", ""))):
                candidate["type"] = "watermark"
                candidate["role"] = "watermark"
            else:
                candidate["type"] = key
            k = _block_key(candidate)
            if k in seen:
                continue
            seen.add(k)
            blocks.append(candidate)
            added += 1

    if shuffle_blocks:
        random.shuffle(blocks)

    # Lấy header/footer từ iterator nếu có, không thì lấy từ layout_config
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

    # Lấy vùng header/footer để tính top/bottom layout
    header_rect = _rect(header_block)
    footer_rect = _rect(footer_block)

    # Tham số layout từ config (có default)
    page_w = float(layout_config.get("page_width",1237))
    left = float(layout_config.get("left_margin", 120))
    right = float(layout_config.get("right_margin", 120))
    row_gap = float(layout_config.get("row_gap", 16))
    col_gap = float(layout_config.get("col_gap", 16))
    margin = float(layout_config.get("header_footer_margin", 8))

    # Vùng đặt content: từ dưới header đến trên footer
    top = (header_rect[1] + header_rect[3] + margin) if header_rect else margin
    bottom = (footer_rect[1] - margin) if footer_rect else float(layout_config.get("page_height", 3508)) - margin

    # Auto gán tọa độ cho các block nội dung
    _layout_blocks._add_separators = bool(layout_config.get("add_separators", False))
    blocks = _layout_blocks(blocks, page_w, top, bottom, left, right, row_gap, col_gap)
    blocks = [header_block] + blocks + [footer_block]

    # In debug ra console và lưu ra file nếu cần
    pretty = pformat(blocks, width=120)
    # print(pretty)
    print("\n" + "*" * 80)
    for b in blocks:
        print(pformat(b, width=120))
        print("*" * 80)
    debug_path = layout_config.get("debug_blocks_path")
    if debug_path:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(pretty + "\n")
    return {"blocks": blocks}
