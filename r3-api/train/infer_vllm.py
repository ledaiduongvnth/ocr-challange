import argparse
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}



SYSTEM_PROMPT = """
You are an OCR assistant. Your task is Extract all information from the document image and return it as a single strictly formatted valid JSON object.
### OUTPUT FORMAT:
- Return ONLY strictly formatted valid JSON object.
- Do NOT output HTML.
- Do NOT wrap in markdown or code blocks.
- Do NOT add explanations.

### CRITICAL JSON FORMATTING
Every item in a dictionary MUST be a valid key-value pair separated by a colon (e.g., `"Key": "Value"`). NEVER output a standalone string without a colon and a value inside an object. 

### STRICT COMMITTEE EXTRACTION GUIDELINES

1. **Document Title:** Always find MAIN DOCUMENT TITLE (tên tài liệu / tên chứng từ) of the WHOLE document if it exists.The title must be in Vietnamese(The title may also contain an English noun) and English translation(if present near the Vietnamese title) is optional. Assign it to the root key `"Title"`. 
   - Note: Landing AI sometimes hides the title inside logo tags (e.g., `<::logo: GIẤY LĨNH TIỀN...::>`). You MUST extract it from there. 
   - HARD RULES (reject candidates that violate any rule):
      + Reject section headings / numbering:
         * title_vi must NOT start with any digit (0-9)
         * title_vi must NOT start with: "A.", "A,", "B.", "B,", "1.", "1,", "I.", "I,", "II.", "II," (case-insensitive)
         * Also reject common section prefixes: "III.", "IV.", "V.", "(1)", "1)", "a)", "-", "•", "*"
      + Reject weird/special characters:
         * Ignore any text containing characters outside Vietnamese letters (including diacritics), spaces, and these punctuation marks only: "-", ",", ".". Title may contains "/", so "/" is allowed.
         * If text contains any of: "_ @ # $ % ^ & * = + < > { } : [ ] ?" then it is NOT a title candidate.
      + Title should be a document name, not a sentence:
         * Must not look like a paragraph (reject if very long or contains many commas)
      + Prefer near to the top-of-page + big font and uppercase:
         * Only consider blocks in the top area of the page (top 25% by position).
         * The best title is usually the block(s) with a big font_size in that top area.
         * If the title spans multiple lines, merge consecutive blocks that are close vertically and have similar font_size.

2. **Noise & Artifact Removal:** - Completely ignore visual noise (watermarks, QR codes, purely visual logos).
   - *Exception:* If any logo or attestation contains actual textual data (like a signature, a stamp stating "ĐÃ CHI TIỀN", or a title), you MUST extract that text.

3. **Sections vs. Root-Level Keys vs. Tables (CRITICAL):**
   - **Floating Key-Value Pairs (Root Level):** If key-value pairs are loosely listed on the page WITHOUT a visual bounding box/square table and WITHOUT a highlighted section name, they MUST remain as flat, root-level keys. Do NOT arbitrarily group them into "Section1".
   - **Form Grids (Sections):** You can ONLY group items into a Section if they are visually enclosed in a drawn square/box OR fall under a clear, highlighted section heading. Group these under their explicit heading. If a valid enclosed box lacks a heading, use `"Section1"`, `"Section2"`.
   - **True Tables:** Only structures with repeated row records under consistent column headers are tables. Name them using the explicit caption above them, or `"Table1"` if unnamed. 
   - **Nesting:** If a table or a free text block appears *inside* a valid form section, nest it as a child of that section.

4. **Table Formatting & Totals:**
   - Format tables as a list of dictionaries. Keys inside the dictionary MUST exactly match the column headers. NEVER hallucinate or invent headers.
   - **Merged Cells:** If a cell is visually merged across multiple columns, duplicate its value into each covered column in the JSON row.
   - **Total Rows (SEPARATE TABLE):** If a standalone "Total" or "Tổng cộng" line appears immediately below a table, DO NOT put it inside the main table. Extract it as a *separate* table (a list containing a single dictionary). Use the exact total label as the root key, and map the values to their corresponding column headers. Example: `"Tổng cộng (Total)": [{"Thành tiền": "71,900,000"}]`.

5. **Signatures & Approvals:** - Extract signature blocks. Use the explicit role/title as the key. 
   - The value MUST include ALL associated text in that block, including printed names, employee IDs (e.g., "YENNT6.NCB"), stamp text, or timestamps. Preserve line breaks (e.g., `"YENNT6.NCB\nNguyễn Thị Yến"`). If no role is found, use `"Signature1"`.

6. **Key-Value & Checkboxes:** - Extract key-value pairs exactly as written. If a value is missing, use an empty string `""`.
   - Represent checkboxes as booleans (e.g., `"Loại tiền (Currency)": {"VND": true, "EUR": false}`).
   - If there are duplicate sibling keys, append `_1`, `_2` to them.

7. **Free Text Aggregation:** - Standalone text blocks (including isolated metadata at the top or bottom of the page like Bank Names, Branch names, or Addresses) MUST be assigned to keys like `"FreeText1"`, `"FreeText2"`. DO NOT use long text blocks as keys without values.
   - Consecutive blocks of free text must be merged into a single `FreeText` string. Preserve line breaks using `\n`. 

8. **BLANK PAGE HANDLING (CRITICAL):**
   - **Blank Page:** If you encounter a completely blank page or a page containing only visual noise/page numbers, ignore it entirely. Do not let it interrupt the extraction. Treat the pages before and after the blank page as perfectly continuous.
   """
USER_PROMPT = (
    "OCR this document image into structured JSON. "
    "Apply all STRICT COMMITTEE EXTRACTION GUIDELINES strictly. "
    "Return ONLY valid JSON."
)



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")

    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _extract_content(resp_json: dict[str, Any]) -> str:
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""

    if isinstance(content, str):
        return content.strip()

    # Some backends can return content blocks.
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(texts).strip()

    return str(content).strip()


def _extract_json_block(text: str) -> str:
    # Remove markdown fence if model wrapped output.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find first plausible JSON block if text has extra chars.
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in [start_obj, start_arr] if i != -1]
    if not starts:
        return text.strip()

    start = min(starts)
    return text[start:].strip()


def _safe_json_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _extract_json_block(text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {"raw_output": text}


def call_vllm(
    image_path: Path,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    api_key: str | None,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    image_data_url = _encode_image_to_data_url(image_path)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
            resp_json = json.loads(body)
    except error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {e.code}: {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"vLLM request failed: {e}") from e

    raw_text = _extract_content(resp_json)
    parsed = _safe_json_parse(raw_text)

    # out = {
    #     "result": parsed,
    #     "meta": {
    #         "model": model,
    #         "finish_reason": (
    #             resp_json.get("choices", [{}])[0].get("finish_reason")
    #             if isinstance(resp_json.get("choices"), list) and resp_json.get("choices")
    #             else None
    #         ),
    #         "usage": resp_json.get("usage"),
    #     },
    # }
    out = parsed
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for merged Chandra model via vLLM OpenAI API")
    parser.add_argument("--input", required=True, help="Input image path or folder")
    parser.add_argument("--output", required=True, help="Output folder for JSON files")
    parser.add_argument("--base-url", default="http://127.0.0.1:7888/v1", help="vLLM base URL")
    parser.add_argument("--model", default="chandra", help="Served model name in vLLM")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    parser.add_argument("--api-key", default=None, help="Optional bearer API key")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

    if not files:
        raise SystemExit(f"No supported images found in: {input_path}")

    logger.info("Processing %d image(s) via %s", len(files), args.base_url)

    for idx, image_path in enumerate(files, 1):
        t0 = time.time()
        logger.info("[%d/%d] %s", idx, len(files), image_path.name)

        try:
            output = call_vllm(
                image_path=image_path,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_sec=args.timeout,
                api_key=args.api_key,
            )
        except Exception as e:
            output = {"result": {"error": str(e)}, "meta": {"model": args.model}}

        out_file = output_dir / f"{image_path.stem}.json"
        out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %s (%.2fs)", out_file.name, time.time() - t0)


if __name__ == "__main__":
    main()
