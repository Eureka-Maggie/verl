# -*- coding: utf-8 -*-
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM client utility for calling external LLM APIs within verl framework."""

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv


class LLMClient:
    """Client for calling external LLM APIs.

    This client is designed to work within the verl framework and supports
    various LLM providers through a unified interface.

    Args:
        token: API token for authentication
        model: Model name to use
        app: Application identifier
        quota_id: Quota identifier
        business_unit: Business unit identifier
        user_id: User identifier
        access_key: Access key for authentication
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 32768)
        env_path: Path to .env file for loading credentials (optional)
    """

    def __init__(
        self,
        token: str = None,
        model: str = None,
        app: str = None,
        quota_id: str = None,
        business_unit: str = None,
        user_id: str = None,
        access_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 32768,
        env_path: str = None,
    ):
        # Load environment variables if env_path is provided
        if env_path and os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)

        self.headers = {
            "content-type": "application/json",
            "token": token or os.getenv("LLM_TOKEN", ""),
        }
        self.url = os.getenv("LLM_API_URL", "https://llm-chat-api.alibaba-inc.com/v1/api/chat")
        self.model = model or os.getenv("LLM_MODEL", "gemini-3-flash-preview")
        self.app = app or os.getenv("LLM_APP", "quark_gen")
        self.business_unit = business_unit or os.getenv("LLM_BUSINESS_UNIT", "")
        self.quota_id = quota_id or os.getenv("LLM_QUOTA_ID", "")
        self.user_id = user_id or os.getenv("LLM_USER_ID", "")
        self.access_key = access_key or os.getenv("LLM_ACCESS_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def extract_content_from_response(response_data: dict) -> Optional[str]:
        """Extract content from API response, handling multiple response formats.

        Args:
            response_data: Raw response dictionary from API

        Returns:
            Extracted content string or None if extraction fails
        """
        if not isinstance(response_data, dict):
            return None
        if response_data.get("code") != 0 and response_data.get("standard_code") != 200:
            return None

        # Try to print cost info if available
        try:
            if "data" in response_data and "cost_info" in response_data["data"]:
                print(f"API cost info: {response_data['data']['cost_info']}")
            if "data" in response_data and "completion" in response_data["data"]:
                print(f"API usage: {response_data['data']['completion']['usage']}")
        except:
            pass

        # Gemini format
        try:
            if "data" in response_data and "data" in response_data["data"] and "message" in response_data["data"]["data"]:
                return response_data["data"]["data"]["message"]
        except (KeyError, TypeError):
            pass

        # Qwen format
        try:
            if "data" in response_data and "completion" in response_data["data"]:
                content = response_data["data"]["completion"]["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    # OpenAI content-parts format: [{"type": "text", "text": "..."}]
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                return content
        except (KeyError, TypeError, IndexError):
            pass

        # data.message format
        try:
            if "data" in response_data and "message" in response_data["data"]:
                return response_data["data"]["message"]
        except (KeyError, TypeError):
            pass

        # data.content format
        try:
            if "data" in response_data and "content" in response_data["data"]:
                return response_data["data"]["content"]
        except (KeyError, TypeError):
            pass

        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = None,
        max_tokens: int = None,
        cache: int = 0,
        tag: str = "chat_request",
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> Dict[str, Any]:
        """Send a chat request to the LLM API with exponential-backoff retry.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (overrides default if provided)
            max_tokens: Maximum tokens to generate (overrides default if provided)
            cache: Cache setting (0 or 1)
            tag: Request tag for tracking
            max_retries: Maximum number of attempts (default 5)
            backoff_base: Base for exponential backoff in seconds (default 2.0,
                          so waits are 2, 4, 8, 16 … seconds between retries)

        Returns:
            Dictionary with 'success' (bool) and either 'content' (str) or 'error'/'response'
        """
        data = {
            "business_unit": self.business_unit,
            "app": self.app,
            "quota_id": self.quota_id,
            "model": self.model,
            "user_id": self.user_id,
            "access_key": self.access_key,
            "prompt": messages,
            "params": {
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            },
            "cache": cache,
            "tag": tag,
        }
        last_result = {"success": False, "error": "No attempts made"}
        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            is_timeout = False
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(data, separators=(",", ":"), ensure_ascii=False),
                    verify=False,
                    timeout=30,
                )
                response.raise_for_status()
                duration = time.time() - start_time
                print(f"[LLMClient] API call took {duration:.2f}s (attempt {attempt}/{max_retries})")
                response_data = response.json()
                content = self.extract_content_from_response(response_data)
                if content is not None:
                    return {"success": True, "content": content}
                # Bad response format — worth retrying
                print(
                    f"[LLMClient] Response format error (attempt {attempt}/{max_retries}): "
                    f"{json.dumps(response_data, ensure_ascii=False, default=str)[:300]}"
                )
                last_result = {"success": False, "response": response_data}
            except requests.exceptions.Timeout:
                duration = time.time() - start_time
                is_timeout = True
                print(
                    f"[LLMClient] Timeout after {duration:.2f}s "
                    f"(attempt {attempt}/{max_retries}), retrying immediately …"
                )
                last_result = {"success": False, "error": "timeout"}
            except requests.exceptions.HTTPError as e:
                duration = time.time() - start_time
                if e.response is not None and e.response.status_code == 429:
                    wait = 8 * attempt  # 8s, 16s, 24s …
                    print(
                        f"[LLMClient] Rate limited (429) after {duration:.2f}s "
                        f"(attempt {attempt}/{max_retries}), waiting {wait}s …"
                    )
                    time.sleep(wait)
                    continue
                err = traceback.format_exc()
                print(f"[LLMClient] HTTP error after {duration:.2f}s (attempt {attempt}/{max_retries}):\n{err}")
                last_result = {"success": False, "error": err}
            except Exception:
                duration = time.time() - start_time
                err = traceback.format_exc()
                print(
                    f"[LLMClient] API call failed after {duration:.2f}s "
                    f"(attempt {attempt}/{max_retries}):\n{err}"
                )
                last_result = {"success": False, "error": err}

            if attempt < max_retries and not is_timeout:
                wait = backoff_base ** (attempt - 1)   # 1s, 2s, 4s, 8s …
                print(f"[LLMClient] Retrying in {wait:.1f}s …")
                time.sleep(wait)

        print(f"[LLMClient] All {max_retries} attempts failed for tag='{tag}'.")
        return last_result

    def simple_text_call(self, system_prompt: str, user_text: str,
                         max_tokens: int = None) -> Tuple[str, bool]:
        """Make a simple text-based call to the LLM.

        Args:
            system_prompt: System prompt to set context
            user_text: User message text
            max_tokens: Maximum tokens to generate (overrides instance default if provided)

        Returns:
            Tuple of (response_text, success_flag)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        result = self.chat(messages, max_tokens=max_tokens)
        if result.get("content") is not None:
            return result["content"], True
        return result.get("response", "") or str(result.get("error", "Call failed")), False