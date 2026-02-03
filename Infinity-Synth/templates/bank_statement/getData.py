import json
import os
import random
from pprint import pformat

from utils.HeaderFooter import produce_header_footer
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
    # Ensure mandatory elements if data exists
    for name in ("header", "footer", "stamp", "signature"):
        if getattr(self, name, None) and elem_cfg.get(name, 0) == 0:
            elem_cfg[name] = 1

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

    for element, max_count in elem_cfg.items():
        if max_count <= 0:
            continue
        if element == "header":
            input_data["header"] = next(self.header_iter)
            continue
        if element == "footer":
            input_data["footer"] = next(self.footer_iter)
            continue
        if element == "page_footnote":
            input_data["page_footnote"] = get_random_text_snippet(self.text_iter)
            continue

        for _ in range(max_count):
            if element == "formula":
                formula = next(self.formula_iter)
                try:
                    formula["latex"] = latextool("$$" + formula["latex"] + "$$")
                except Exception:
                    continue
                add_unique(formula)
                continue

            iter_name = iter_map.get(element)
            if iter_name:
                add_unique(next(getattr(self, iter_name)))

    if layout_config.get("shuffle", True):
        random.shuffle(column)

    input_data["body"] = column
    if len(column) < 2:
        return None 

    title = None

    for dat in column:
        if dat.get("type") == "Body":
            title = dat.get("heading")

    if "header" not in input_data and getattr(self, "header", None):
        input_data["header"] = next(self.header_iter)
    if "footer" not in input_data and getattr(self, "footer", None):
        input_data["footer"] = next(self.footer_iter)
    if title and "header" not in input_data and "footer" not in input_data:
        head_foot = produce_header_footer(title)
        input_data["header"] = head_foot.get("header")
        input_data["footer"] = head_foot.get("footer")

    debug_path = layout_config.get("debug_blocks_path")
    if debug_path:
        try:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(pformat(input_data, width=120))
        except OSError:
            pass

    return input_data
