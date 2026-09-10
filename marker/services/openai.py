import json
import re
import time
from typing import Annotated, List

import openai
import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel, ValidationError

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()

# Matches a trailing comma before a closing brace/bracket, e.g. `"a": 1,}`.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _repair_json(text: str) -> dict:
    """Best-effort repair for near-valid JSON some non-OpenAI models emit
    behind an OpenAI-compatible endpoint (e.g. a trailing comma), since
    strict client-side validation otherwise rejects an output that's
    almost certainly still usable."""
    return json.loads(_TRAILING_COMMA_RE.sub(r"\1", text))


class OpenAIService(BaseService):
    openai_base_url: Annotated[
        str, "The base url to use for OpenAI-like models.  No trailing slash."
    ] = "https://api.openai.com/v1"
    openai_model: Annotated[str, "The model name to use for OpenAI-like model."] = (
        "gpt-5-mini"
    )
    openai_api_key: Annotated[
        str, "The API key to use for the OpenAI-like service."
    ] = None
    openai_image_format: Annotated[
        str,
        "The image format to use for the OpenAI-like service. Use 'png' for better compatability",
    ] = "webp"

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        Generate the base-64 encoded message to send to an
        openAI-compatabile multimodal model.

        Args:
            images: Image or list of PIL images to include
            format: Format to use for the image; use "png" for better compatability.

        Returns:
            A list of OpenAI-compatbile multimodal messages containing the base64-encoded images.
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.openai_image_format
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/{};base64,{}".format(
                        img_fmt, self.img_to_base64(img, format=img_fmt)
                    ),
                },
            }
            for img in images
        ]

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

        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                # parse() is GA in openai>=1.92 - use it directly (not beta)
                response = client.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                        "HTTP-Referer": "https://github.com/datalab-to/marker",
                    },
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                response_text = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                # Rate limit exceeded
                if tries == total_tries:
                    # Last attempt failed. Give up
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                    )
                    time.sleep(wait_time)
            except ValidationError as e:
                # Some non-OpenAI models behind an OpenAI-compatible endpoint
                # don't fully honor strict JSON-schema decoding and can
                # deterministically emit near-valid JSON (e.g. a trailing
                # comma), so a blind retry just repeats the same bad output.
                # Try to repair the raw text pydantic captured before falling
                # back to a retry.
                raw = e.errors()[0].get("input") if e.errors() else None
                if isinstance(raw, str):
                    try:
                        repaired = _repair_json(raw)
                        logger.warning(f"Repaired malformed JSON from model output: {e}")
                        return repaired
                    except json.JSONDecodeError:
                        pass

                if tries == total_tries:
                    logger.error(
                        f"Invalid JSON response: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    logger.warning(
                        f"Invalid JSON response: {e}. Retrying... (Attempt {tries}/{total_tries})",
                    )
            except json.JSONDecodeError as e:
                if tries == total_tries:
                    logger.error(
                        f"Invalid JSON response: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    logger.warning(
                        f"Invalid JSON response: {e}. Retrying... (Attempt {tries}/{total_tries})",
                    )
            except Exception as e:
                logger.error(f"OpenAI inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
