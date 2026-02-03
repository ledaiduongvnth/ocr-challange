import json
import os
import random
from pprint import pformat

from utils.LatexUtil import LatexNormalizer
from utils.utils import get_random_text_snippet

latextool = LatexNormalizer()


def get_data(self, layout_config):
    input_data = {}
    column = []
    seen = set()

    def add_unique(item):
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            return
        seen.add(key)
        column.append(item)

    elem_cfg = dict(layout_config.get("element", {}))
    iter_map = {
        "title": "title_iter",
        "text": "text_iter",
        "table": "table_iter",
        "figure": "figure_iter",
        "image": "figure_iter",
        "logo": "logo_iter",
        "form": "form_iter",
        "stamp": "stamp_iter",
        "signature": "signature_iter",
    }

    def next_or_none(it):
        try:
            return next(it)
        except StopIteration:
            return None

    def next_required(it, element):
        item = next_or_none(it)
        if item is None:
            raise ValueError(f"Missing data for required element: {element}")
        return item

    def next_item_required(element):
        iter_name = iter_map.get(element)
        if not iter_name:
            return None
        return next_required(getattr(self, iter_name), element)

    for element, max_count in elem_cfg.items():
        if max_count <= 0:
            continue
        if element == "header":
            input_data["header"] = next_required(self.header_iter, "header")
            continue
        if element == "footer":
            input_data["footer"] = next_required(self.footer_iter, "footer")
            continue
        if element == "page_footnote":
            input_data["page_footnote"] = get_random_text_snippet(self.text_iter)
            continue

        for _ in range(max_count):
            if element == "formula":
                formula = next_required(self.formula_iter, "formula")
                try:
                    formula["latex"] = latextool("$$" + formula["latex"] + "$$")
                except Exception:
                    continue
                add_unique(formula)
                continue

            item = next_item_required(element)
            if item is not None:
                add_unique(item)

    if layout_config.get("shuffle", True):
        random.shuffle(column)

    input_data["body"] = column

    if elem_cfg.get("header", 0) > 0 and "header" not in input_data and getattr(self, "header", None):
        input_data["header"] = next_required(self.header_iter, "header")
    if elem_cfg.get("footer", 0) > 0 and "footer" not in input_data and getattr(self, "footer", None):
        input_data["footer"] = next_required(self.footer_iter, "footer")

    debug_path = layout_config.get("debug_blocks_path")
    if debug_path:
        try:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            input_data["_layout_order"] = [item.get("type") for item in column]
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(pformat(input_data, width=120))
        except OSError:
            pass

    return input_data
