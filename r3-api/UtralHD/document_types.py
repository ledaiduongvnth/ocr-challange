import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict

DEFAULT_DOCUMENT_TYPE_CODE = "OTHER"

# Canonical document class codes and their original Vietnamese labels.
# Response label format: CODE only (for example: LIET_KE_GIAO_DICH)
DOCUMENT_TYPE_CLASSES: Dict[str, dict[str, str]] = {
    "LIET_KE_GIAO_DICH": {
        "vi": "Liệt kê giao dịch",
        "response": "LIET_KE_GIAO_DICH",
    },
    "SO_QUY": {
        "vi": "Sổ quỹ",
        "response": "SO_QUY",
    },
    "GIAY_DE_NGHI_TIEP_QUY": {
        "vi": "Giấy đề nghị tiếp quỹ",
        "response": "GIAY_DE_NGHI_TIEP_QUY",
    },
    "GIAY_RUT_TIEN": {
        "vi": "Giấy rút tiền",
        "response": "GIAY_RUT_TIEN",
    },
    "GIAY_GUI_TIEN_TIET_KIEM": {
        "vi": "Giấy gửi tiền tiết kiệm",
        "response": "GIAY_GUI_TIEN_TIET_KIEM",
    },
    "PHIEU_HACH_TOAN": {
        "vi": "Phiếu hạch toán",
        "response": "PHIEU_HACH_TOAN",
    },
    "GIAY_DE_NGHI_SU_DUNG_DICH_VU_INTERNET_BANKING": {
        "vi": "Giấy đề nghị sử dụng dịch vụ internet banking",
        "response": "GIAY_DE_NGHI_SU_DUNG_DICH_VU_INTERNET_BANKING",
    },
    "GIAY_DE_NGHI_THAY_DOI_THONG_TIN_DICH_VU_INTERNET_BANKING": {
        "vi": "Giấy đề nghị thay đổi thông tin dịch vụ internet banking",
        "response": "GIAY_DE_NGHI_THAY_DOI_THONG_TIN_DICH_VU_INTERNET_BANKING",
    },
    "TO_TRINH_THAM_DINH_TIN_DUNG": {
        "vi": "Tờ trình thẩm định tín dụng",
        "response": "TO_TRINH_THAM_DINH_TIN_DUNG",
    },
    "CCCD": {
        "vi": "Căn cước công dân",
        "response": "CCCD",
    },
    "LENH_CHUYEN_TIEN": {
        "vi": "Lệnh chuyển tiền",
        "response": "LENH_CHUYEN_TIEN",
    },
    "GIAY_PHONG_TOA_TAM_KHOA_TAI_KHOAN": {
        "vi": "Giấy phong tỏa/tạm khoá tài khoản",
        "response": "GIAY_PHONG_TOA_TAM_KHOA_TAI_KHOAN",
    },
    "OTHER": {
        "vi": "Khác",
        "response": "OTHER",
    },
}

# Optional human-readable aliases used by OCR output or short-form titles (no acronym aliases).
DOCUMENT_TYPE_ALIASES: Dict[str, list[str]] = {
    # OTHER intentionally has no aliases: it is only fallback for unknown types.
    "LIET_KE_GIAO_DICH": ["Sao ke giao dich", "Liet ke giao dich"],
    "SO_QUY": ["So quy tien mat", "So quy tien", "So quy"],
    "GIAY_DE_NGHI_TIEP_QUY": ["De nghi tiep quy", "Yeu cau tiep quy"],
    "GIAY_RUT_TIEN": ["Cash withdrawal slip", "Rut tien"],
    "GIAY_GUI_TIEN_TIET_KIEM": ["So tiet kiem", "Gui tiet kiem"],
    "PHIEU_HACH_TOAN": ["Phieu hach toan", "Chung tu hach toan"],
    "GIAY_DE_NGHI_SU_DUNG_DICH_VU_INTERNET_BANKING": ["Dang ky internet banking", "Su dung internet banking"],
    "GIAY_DE_NGHI_THAY_DOI_THONG_TIN_DICH_VU_INTERNET_BANKING": ["Thay doi internet banking", "Cap nhat internet banking"],
    "TO_TRINH_THAM_DINH_TIN_DUNG": ["To trinh tham dinh tin dung", "Tham dinh tin dung"],
    "CCCD": ["Can cuoc", "Can cuoc cong dan", "Chung minh nhan dan", "CMND"],
    "LENH_CHUYEN_TIEN": ["Chuyen tien"],
    "GIAY_PHONG_TOA_TAM_KHOA_TAI_KHOAN": ["Phong toa tai khoan", "Tam khoa tai khoan", "Phong toa tam khoa"],
}


def normalize_document_type_key(value: str | None) -> str:
    """Normalize text to a code-like token for robust matching."""
    if not value:
        return ""

    ascii_text = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in ascii_text if not unicodedata.combining(ch))
    ascii_text = ascii_text.replace("đ", "d").replace("Đ", "D")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return ascii_text.upper()


def _token_set(normalized_value: str) -> set[str]:
    return {token for token in normalized_value.split("_") if token}


def _combined_similarity(query: str, candidate: str) -> float:
    """Blend fuzzy and token-based scores to handle OCR noise."""
    ratio = SequenceMatcher(None, query, candidate).ratio()
    q_tokens = _token_set(query)
    c_tokens = _token_set(candidate)

    if not q_tokens or not c_tokens:
        token_overlap = 0.0
        query_coverage = 0.0
    else:
        intersection = len(q_tokens & c_tokens)
        token_overlap = intersection / len(c_tokens)
        query_coverage = intersection / len(q_tokens)

    contains_score = 0.9 if (query in candidate or candidate in query) else 0.0
    blended = (0.60 * ratio) + (0.25 * token_overlap) + (0.15 * query_coverage)
    return max(contains_score, blended)


NORMALIZED_DOCUMENT_TYPE_LOOKUP: Dict[str, str] = {}
NORMALIZED_CANDIDATES_BY_CODE: Dict[str, list[str]] = {}
for code, info in DOCUMENT_TYPE_CLASSES.items():
    variants = [code, info["vi"], info["response"], *DOCUMENT_TYPE_ALIASES.get(code, [])]

    code_candidates: list[str] = []
    for variant in variants:
        normalized_variant = normalize_document_type_key(variant)
        if not normalized_variant:
            continue

        NORMALIZED_DOCUMENT_TYPE_LOOKUP[normalized_variant] = code
        code_candidates.append(normalized_variant)

    NORMALIZED_CANDIDATES_BY_CODE[code] = code_candidates


def _is_token_boundary_match(text: str, start_idx: int, candidate_len: int) -> bool:
    end_idx = start_idx + candidate_len
    left_ok = start_idx == 0 or text[start_idx - 1] == "_"
    right_ok = end_idx == len(text) or text[end_idx] == "_"
    return left_ok and right_ok


def _match_by_front_position(normalized_value: str) -> str | None:
    """Pick the document type whose alias/label appears earliest in title text."""
    best_code: str | None = None
    best_start = len(normalized_value) + 1
    best_len = -1

    for code, candidates in NORMALIZED_CANDIDATES_BY_CODE.items():
        if code == DEFAULT_DOCUMENT_TYPE_CODE:
            continue

        for candidate in candidates:
            if not candidate:
                continue

            start_idx = normalized_value.find(candidate)
            while start_idx != -1:
                if _is_token_boundary_match(normalized_value, start_idx, len(candidate)):
                    if (start_idx < best_start) or (start_idx == best_start and len(candidate) > best_len):
                        best_start = start_idx
                        best_len = len(candidate)
                        best_code = code
                    break

                start_idx = normalized_value.find(candidate, start_idx + 1)

    return best_code


def match_document_type_code(raw_value: str | None, soft_threshold: float = 0.58) -> str:
    """Map raw title to canonical code using exact match first, then soft matching."""
    normalized_value = normalize_document_type_key(raw_value)
    if not normalized_value:
        return DEFAULT_DOCUMENT_TYPE_CODE

    # Hard match first: code, Vietnamese label, response label, aliases.
    hard_match = NORMALIZED_DOCUMENT_TYPE_LOOKUP.get(normalized_value)
    if hard_match:
        return hard_match

    # If multiple type aliases exist in title text, choose the type whose alias appears first.
    front_position_match = _match_by_front_position(normalized_value)
    if front_position_match:
        return front_position_match

    # Soft match: choose the best score across all document classes.
    best_code = DEFAULT_DOCUMENT_TYPE_CODE
    best_score = 0.0
    for code, candidates in NORMALIZED_CANDIDATES_BY_CODE.items():
        if code == DEFAULT_DOCUMENT_TYPE_CODE or not candidates:
            continue

        score = max(_combined_similarity(normalized_value, candidate) for candidate in candidates)
        if score > best_score:
            best_score = score
            best_code = code

    if best_score >= soft_threshold:
        return best_code

    return DEFAULT_DOCUMENT_TYPE_CODE
