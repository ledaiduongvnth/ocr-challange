# IDP Document Classification & Extraction API

This project provides a FastAPI service for cache-based document classification and extraction.

Current implementation is offline and deterministic:
- `/classification` reads JSON files from `json_files_cache`, applies document type matching rules, and returns grouped page classification.
- `/extract` reads one cached JSON file by uploaded filename stem and returns it as response data.

## 📁 Project Structure

    project_root/
    ├── Dockerfile
    ├── requirements.txt
    ├── app.py
    ├── app_utils.py
    ├── document_types.py
    └── json_files_cache/
        ├── split_document_page_0.json
        ├── split_document_page_1.json
        └── ...

## ⚙️ Configuration

- `JSON_FILES_CACHE_DIR` (optional): absolute path to cache directory.
- Default value:
  `/media/drive-2t/hoangnv83/code/ocr/ocr-challange-duong/r3-api/UtralHD/json_files_cache`

## 🚀 1. Build and Run the Docker Container

**Step 1: Build image**

    docker build -t idp_api .

**Step 2: Run container**

    docker run -p 47924:8000 \
      -e JSON_FILES_CACHE_DIR="/app/json_files_cache" \
      idp_api

If you use the default cache path inside your environment, you can omit `JSON_FILES_CACHE_DIR`.

---

## 🧪 2. How to Test and Use the API

Once running, API base URL is:
`http://localhost:47924`

### Option A: Swagger UI

Open:
`http://localhost:47924/docs`

### Option B: API Endpoints & cURL Examples

#### Endpoint 1: `/classification`
- **Method:** `POST`
- **Input:** multipart form-data
  - `file`: any uploaded file (kept for API compatibility).
- **Description:**
  - Reads all `*.json` files in cache directory.
  - Extracts title/document type from JSON keys: `document_type`, `DocumentType`, `Title`, `title`.
  - Applies hard + soft matching to predefined classes.
  - Uses title pages as separators and groups consecutive untitled pages into nearest previous titled group.

**cURL Example:**

    curl -X POST "http://localhost:47924/classification" \
      -H "accept: application/json" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@test_document.pdf"

**Expected Output (example):**

    {
      "status": "success",
      "page_count": 5,
      "classification": [
        {
          "index": 0,
          "document_type": "GIAY_RUT_TIEN",
          "pages": [0]
        },
        {
          "index": 1,
          "document_type": "CCCD",
          "pages": [1, 2]
        }
      ]
    }

#### Endpoint 2: `/extract`
- **Method:** `POST`
- **Input:** multipart form-data
  - `file`: uploaded document page (PDF/Image)
  - `document_type`: request field for client compatibility (not used for extraction logic)
- **Description:**
  - Maps uploaded filename stem to cache JSON filename.
  - Example: `split_document_page_0.pdf` -> `split_document_page_0.json`.
  - Returns cached JSON as `data`.
  - Deletes that cache JSON file at end of request.

**cURL Example:**

    curl -X POST "http://localhost:47924/extract" \
      -H "accept: application/json" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@split_document_page_0.pdf" \
      -F "document_type=GIAY_RUT_TIEN"

**Expected Output (example):**

    {
      "status": "success",
      "data": {
        "Title": "GIẤY RÚT TIỀN",
        "Tên tài khoản": "NGUYEN VIET HUNG",
        "Số tài khoản": "100013014668"
      }
    }

## 🧾 Supported Document Type Codes

- `LIET_KE_GIAO_DICH`
- `SO_QUY`
- `GIAY_DE_NGHI_TIEP_QUY`
- `GIAY_RUT_TIEN`
- `GIAY_GUI_TIEN_TIET_KIEM`
- `PHIEU_HACH_TOAN`
- `GIAY_DE_NGHI_SU_DUNG_DICH_VU_INTERNET_BANKING`
- `GIAY_DE_NGHI_THAY_DOI_THONG_TIN_DICH_VU_INTERNET_BANKING`
- `TO_TRINH_THAM_DINH_TIN_DUNG`
- `CCCD`
- `LENH_CHUYEN_TIEN`
- `GIAY_PHONG_TOA_TAM_KHOA_TAI_KHOAN`
- `OTHER`

## 🛠️ Error Handling

Standard error response:

    {
      "status": "error",
      "message": "Description of the error..."
    }

Common cases:
- Cache JSON file not found for `/extract` -> `404`
- Invalid cache JSON format -> `500`
- Invalid upload filename -> `400`
