#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def parse_style(style_str: str):
    style = {}
    if not style_str:
        return style
    for part in style_str.split(';'):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            continue
        k, v = part.split(':', 1)
        style[k.strip()] = v.strip()
    return style


def px(value: str):
    if value is None:
        return None
    m = re.match(r"(-?\d+(?:\.\d+)?)px", value.strip())
    if not m:
        return None
    return float(m.group(1))


def style_without_pos(style: dict, keys):
    parts = []
    for k, v in style.items():
        if k in keys:
            continue
        parts.append(f"{k}: {v}")
    return "; ".join(parts) + (";" if parts else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to preview_styled_*.html")
    parser.add_argument("--output", required=True, help="Path to blocks.json")
    args = parser.parse_args()

    html = Path(args.input).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    page = soup.select_one(".page")
    if page is None:
        raise SystemExit("No .page element found")

    blocks = []
    for child in page.children:
        if getattr(child, "name", None) is None:
            continue

        if child.name == "img":
            style = parse_style(child.get("style", ""))
            x = px(style.get("left", "0px")) or 0.0
            y = px(style.get("top", "0px")) or 0.0
            w = px(style.get("width", "0px")) or 0.0
            h = px(style.get("height", "0px")) or 0.0
            blocks.append({
                "type": "image",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "src": child.get("src", ""),
                "style": style_without_pos(style, {"left", "top", "width", "height"})
            })
            continue

        if "block" in (child.get("class") or []):
            style = parse_style(child.get("style", ""))
            x = px(style.get("left", "0px")) or 0.0
            y = px(style.get("top", "0px")) or 0.0
            w = px(style.get("width", "0px")) or 0.0
            h = px(style.get("height", "0px")) or 0.0

            header = None
            header_boxed = False
            boxed = child.select_one(".boxed-header")
            if boxed:
                header = boxed.get_text(strip=True)
                header_boxed = True
            else:
                h_el = child.select_one(".header")
                if h_el:
                    header = h_el.get_text(strip=True)

            content_el = child.select_one(".content")
            content_html = content_el.decode_contents() if content_el else ""

            block_type = "table" if "table" in (child.get("class") or []) else "block"

            blocks.append({
                "type": block_type,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "header": header or "",
                "header_boxed": header_boxed,
                "content": content_html.strip(),
            })

    Path(args.output).write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
