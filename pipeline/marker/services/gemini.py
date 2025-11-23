import base64
import json
import time
import traceback
from io import BytesIO
from typing import List, Annotated, Optional

import PIL
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from pydantic import BaseModel

from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()

class GoogleGeminiService(BaseService):
    """
    HIJACKED SERVICE:
    """
    gemini_api_key: Annotated[str, "Fake Key"] = "ignored"
    
# VLLM_API_KEY = ""
# VLLM_MODEL_NAME = "chandra"
    _real_base_url: str = ""
    _real_api_key: str = ""
    _real_model: str = "chandra"

    def get_client(self, timeout: int):
        return OpenAI(
            api_key=self._real_api_key,
            base_url=self._real_base_url,
            timeout=timeout,
        )

    def encode_image(self, image: PIL.Image.Image) -> str:
        buffered = BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        timeout = 300 

        client = self.get_client(timeout=timeout)

        try:
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            full_prompt = f"{prompt}\n\nIMPORTANT: You must output strictly valid JSON matching this schema:\n{schema_json}"
        except Exception:
            full_prompt = prompt

        content = [{"type": "text", "text": full_prompt}]

        if image:
            images = image if isinstance(image, list) else [image]
            for img in images:
                base64_image = self.encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts information into JSON format."},
            {"role": "user", "content": content}
        ]

        total_tries = max_retries + 1
        temperature = 0.1 

        for tries in range(1, total_tries + 1):
            # try:
                response = client.chat.completions.create(
                    model=self._real_model,
                    messages=messages,
                    max_tokens=4096, 
                    temperature=temperature,
                    response_format={"type": "json_object"} 
                )

                output_text = response.choices[0].message.content
                
                # Tracking metadata
                if block and response.usage:
                    block.update_metadata(
                        llm_tokens_used=response.usage.total_tokens, 
                        llm_request_count=1
                    )

                # Parse JSON
                return json.loads(output_text)

            # except (APIConnectionError, RateLimitError, APIStatusError) as e:
            #     if tries == total_tries:
            #         logger.error(f"HF Space Connection Error: {e}. Giving up.")
            #         break
            #     wait_time = tries * self.retry_wait_time
            #     logger.warning(f"Connection Error: {e}. Retrying in {wait_time}s...")
            #     time.sleep(wait_time)

            # except json.JSONDecodeError as e:
            #     temperature = min(temperature + 0.2, 1.0)
            #     if tries == total_tries:
            #         logger.error(f"JSON Decode Error: {e}. Last Output: {output_text[:100]}...")
            #         break
            #     logger.warning(f"JSON Error. Retrying with temp={temperature}...")
            
            # except Exception as e:
            #     logger.error(f"Unexpected Error in Chandra Service: {e}")
            #     traceback.print_exc()
            #     break

        return {}