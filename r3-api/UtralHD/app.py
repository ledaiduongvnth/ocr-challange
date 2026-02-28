import os
import time
import json
import uuid
import logging
import mimetypes
import fitz  # PyMuPDF
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from google import genai
from google.genai import types
import json_repair

from landingai_ade import LandingAIADE

# --- LOGGING CONFIGURATION ---
# Standard convention for APIs: Timestamps, Log Level, and Message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("IDP_API")

mimetypes.add_type('image/webp', '.webp')

# --- CONFIGURATION ---
LANDING_AI_API_KEY = os.environ.get("LANDING_AI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
JSON_FILES_CACHE_DIR = Path("/media/hdd01/PycharmProjects/ocr-challange/r3-api/UtralHD/json_files_cache")

landing_client = LandingAIADE(apikey=LANDING_AI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="IDP Extraction & Classification API")

FEW_SHOT_EXAMPLES = [
    {
        "image_path": "./prompts/example1_image.webp",
        "ocr_path": "./prompts/example1_ocr.txt",
        "json_path": "./prompts/example1_json.json"
    },
    {
        "image_path": "./prompts/example2_image.webp",
        "ocr_path": "./prompts/example2_ocr.txt",
        "json_path": "./prompts/example2_json.json"
    }
]


def load_static_classification() -> list:
    classification = []
    for index, json_path in enumerate(sorted(JSON_FILES_CACHE_DIR.glob("*.json"))):
        document_type = None
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                title = data.get("Title", data.get("title"))
                if isinstance(title, str) and title.strip():
                    document_type = title
        except Exception as e:
            logger.warning(f"[-] Failed to read {json_path.name}: {e}")

        classification.append({
            "index": index,
            "document_type": document_type,
            "pages": [index]
        })

    return classification


STATIC_CLASSIFICATION = load_static_classification()

# --- HELPER FUNCTIONS ---
def get_page_count(file_path: str) -> int:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == 'application/pdf':
        try:
            with fitz.open(file_path) as doc:
                return len(doc)
        except Exception as e:
            logger.warning(f"[-] Could not read PDF page count: {e}. Defaulting to 1.")
            return 1
    return 1

def upload_to_gemini(file_path: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            logger.info(f"[*] Uploading {file_path} to Gemini (Attempt {attempt + 1}/{max_retries})...")
            uploaded_file = gemini_client.files.upload(file=file_path)
            
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(3)
                uploaded_file = gemini_client.files.get(name=uploaded_file.name)
                
            if uploaded_file.state.name == "FAILED":
                raise Exception("Gemini server failed to process the file.")
                
            logger.info(f"[+] Uploaded successfully: {uploaded_file.name}")
            return uploaded_file
            
        except Exception as e:
            logger.error(f"[-] Network drop or error: {e}")
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                logger.info(f"[*] Reconnecting in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"[FAILED] Failed to upload {file_path} after {max_retries} attempts.")
                raise e

def extract_markdown_with_landing_ai(file_path: str) -> str:
    logger.info(f"[*] Sending {file_path} to Landing AI (dpt-2-latest) for OCR...")
    response = landing_client.parse(document=Path(file_path), model="dpt-2-latest")
    logger.info("[+] OCR Extraction Complete.")
    return response.markdown


# --- ENDPOINT 1: CLASSIFICATION ---
@app.post("/classification")
async def classification_endpoint(file: UploadFile = File(...)):
    logger.info(f"========== NEW CLASSIFICATION REQUEST: {file.filename} ==========")
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"[*] Saved temporary file to {temp_path}")
        class_array = STATIC_CLASSIFICATION
        page_count = len(class_array)
        logger.info(f"[*] Returning static classification from cache ({page_count} pages).")

        logger.info("[SUCCESS] Classification completed successfully.")
        return JSONResponse(content={
            "status": "success",
            "page_count": page_count,
            "classification": class_array
        })

    except Exception as e:
        logger.error(f"[FAILED] Error processing classification for {file.filename}: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[*] Cleaned up temporary file {temp_path}")


# --- ENDPOINT 2: EXTRACT ---
@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...), document_type: str = Form(...)):
    logger.info(f"========== NEW EXTRACTION REQUEST: {file.filename} | TYPE: {document_type} ==========")
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"[*] Saved temporary file to {temp_path}")
            
        target_ocr = extract_markdown_with_landing_ai(temp_path)
        
        with open("./prompts/system_rules.txt", "r", encoding="utf-8") as f:
            system_rules = f.read()

        logger.info("[*] Preparing Multimodal Prompt with Few-Shot Examples...")
        prompt_contents = []
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
            logger.info(f"[*] Loading Example {i} files...")
            with open(ex["ocr_path"], "r", encoding="utf-8") as f:
                ex_ocr = f.read()
            with open(ex["json_path"], "r", encoding="utf-8") as f:
                ex_json = f.read()
            gemini_ex_file = upload_to_gemini(ex["image_path"])
            prompt_contents.extend([
                f"\n### EXAMPLE {i} ###\nImage:", gemini_ex_file,
                f"\nOCR:\n{ex_ocr}\nTarget JSON:\n{ex_json}"
            ])

        logger.info("[*] Loading Target files...")
        gemini_target_file = upload_to_gemini(temp_path)
        prompt_contents.extend([
            f"\n### YOUR TURN ###\nThe committee classified this document as: {document_type}.",
            "Process this target document:", gemini_target_file,
            f"\nTarget OCR:\n{target_ocr}\nReturn strict JSON."
        ])

        logger.info("[*] Generating JSON structure with Gemini 2.5 Flash...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_rules,
                response_mime_type="application/json", 
                temperature=0.1
            )
        )
        
        logger.info("[*] Applying Auto-Repair to JSON output...")
        parsed_result = json_repair.repair_json(response.text, return_objects=True)
        
        logger.info("[SUCCESS] Extraction completed successfully.")
        return JSONResponse(content={
            "status": "success",
            "data": parsed_result
        })

    except Exception as e:
        logger.error(f"[FAILED] Error processing extraction for {file.filename}: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[*] Cleaned up temporary file {temp_path}")
