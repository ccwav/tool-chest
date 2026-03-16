"""
title: Google Gemini Pipeline
author: ccwav
version: 2.5
required_open_webui_version: 0.8.0
license: Apache License 2.0
"""

import os
import re
import time
import asyncio
import base64
import hashlib
import logging
import io
import uuid
import threading
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
from typing import List, Union, Optional, Dict, Any, Tuple, AsyncIterator, Callable
from pydantic_core import core_schema
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from open_webui.env import SRC_LOG_LEVELS
from fastapi import Request
from open_webui.models.users import UserModel, Users


# Simplified encryption implementation with automatic handling
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """
        Generate encryption key from WEBUI_SECRET_KEY if available
        Returns None if no key is configured
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None

        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """
        Encrypt a string value if a key is available
        Returns the original value if no key is available
        """
        if not value or value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No encryption if no key
            return value

        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """
        Decrypt an encrypted string value if a key is available
        Returns the original value if no key is available or decryption fails
        """
        if not value or not value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No decryption if no key
            return value[len("encrypted:") :]  # Return without prefix

        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    # Pydantic integration
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


class Pipe:
    """
    Pipeline for interacting with Google Gemini models.
    """

    # Configuration valves for the pipeline
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default=os.getenv(
                "GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/"
            ),
            description="Google Generative AI API 的基础 URL。",
        )
        GOOGLE_API_KEY: EncryptedStr = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="Google Generative AI 的 API Key。支持使用英文逗号分隔多个 Key，并按轮询方式选择。遇到额度/限流错误会自动切换到下一个 Key。",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_VERSION: str = Field(
            default=os.getenv("GOOGLE_API_VERSION", "v1beta"),
            description="Google Generative AI 使用的 API 版本（例如 v1alpha、v1beta、v1）。",
        )
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_STREAMING_ENABLED", "true").lower() == "true",
            description="是否启用流式响应（设为 false 可强制使用非流式模式）。",
        )
        STREAM_START_TIMEOUT_SEC: float = Field(
            default=float(os.getenv("GOOGLE_STREAM_START_TIMEOUT_SEC", "20")),
            description="流式请求建立阶段的超时时间（秒）。<=0 表示不限制。",
        )
        INCLUDE_THOUGHTS: bool = Field(
            default=os.getenv("GOOGLE_INCLUDE_THOUGHTS", "true").lower() == "true",
            description="是否启用 Gemini 思考过程输出（设为 false 则关闭）。",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("GOOGLE_THINKING_BUDGET", "-1")),
            description="Gemini 2.5 模型的思考预算（0=禁用，-1=动态，1-32768=固定 token 上限）。"
            "Gemini 3 模型不使用该参数，而使用 THINKING_LEVEL。",
        )
        THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_THINKING_LEVEL", ""),
            description="Gemini 3 模型的思考等级（'minimal'、'low'、'medium'、'high'）。"
            "其他模型会忽略该参数。空字符串表示使用模型默认值。",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower()
            == "true",
            description="是否在内容生成中使用更宽松的安全策略。",
        )
        MODEL_LIST: str = Field(
            default=os.getenv(
                "GOOGLE_MODEL_LIST",
                "gemini-2.5-flash,gemini-3-flash-preview,gemini-3.1-flash-lite-preview",
            ),
            description="手动指定可用模型列表（英文逗号分隔）。"
            "支持 `模型ID` 或 `模型ID:显示名称` 格式。",
        )
        RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_RETRY_COUNT", "2")),
            description="API 调用在临时失败时的重试次数。",
        )
        EMPTY_RESPONSE_RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_EMPTY_RESPONSE_RETRY_COUNT", "2")),
            description="空回复自动重试次数（独立于 RETRY_COUNT）。"
            "0 表示关闭，正整数表示重试次数。"
            "兼容旧配置：-1 会回退为 RETRY_COUNT。",
        )
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default=os.getenv("GOOGLE_DEFAULT_SYSTEM_PROMPT", ""),
            description="应用到所有聊天的默认系统提示词。若用户定义了系统提示词，则会将本提示词前置。留空可禁用。",
        )
        ENABLE_FORWARD_USER_INFO_HEADERS: bool = Field(
            default=os.getenv(
                "GOOGLE_ENABLE_FORWARD_USER_INFO_HEADERS", "false"
            ).lower()
            == "true",
            description="是否转发用户信息请求头。",
        )
        ENABLE_GEMINI_25_FLASH_SEARCH_MODEL: bool = Field(
            default=os.getenv(
                "GOOGLE_ENABLE_GEMINI_25_FLASH_SEARCH_MODEL", "false"
            ).lower()
            == "true",
            description="是否增加 gemini-2.5-flash-search 模型（启用后该模型默认携带内置 Google Search 工具）。",
        )
        # Image Processing Configuration
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", "15.0")),
            description="触发压缩前允许的最大图片大小（MB）。",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", "2048")),
            description="触发缩放前允许的最大宽或高（像素）。",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", "85")),
            description="JPEG 压缩质量（1-100，越高质量越好但体积更大）。",
        )
        IMAGE_ENABLE_OPTIMIZATION: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ENABLE_OPTIMIZATION", "true").lower()
            == "true",
            description="是否启用智能图像优化以提高 API 兼容性。",
        )
        IMAGE_PNG_COMPRESSION_THRESHOLD_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_PNG_THRESHOLD_MB", "0.5")),
            description="当 PNG 文件大于该大小（MB）时，将转换为 JPEG 以获得更高压缩率。",
        )
        IMAGE_HISTORY_MAX_REFERENCES: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_HISTORY_MAX_REFERENCES", "5")),
            description="单次生成请求中可包含的最大图片总数（历史 + 当前消息）。",
        )
        IMAGE_ADD_LABELS: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ADD_LABELS", "true").lower() == "true",
            description="若为 true，会在每张图片前添加如 [Image 1] 的文本标签，便于模型引用。",
        )
        IMAGE_DEDUP_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_DEDUP_HISTORY", "true").lower() == "true",
            description="若为 true，构建历史上下文时会按哈希去重相同图片。",
        )
        IMAGE_HISTORY_FIRST: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_HISTORY_FIRST", "true").lower() == "true",
            description="若为 true（默认），先放历史图片再放当前消息图片；若为 false，则相反。",
        )

    # ---------------- Internal Helpers ---------------- #
    async def _gather_history_images(
        self,
        messages: List[Dict[str, Any]],
        last_user_msg: Dict[str, Any],
        optimization_stats: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        history_images: List[Dict[str, Any]] = []
        for msg in messages:
            if msg is last_user_msg:
                continue
            if msg.get("role") not in {"user", "assistant"}:
                continue
            _p, parts = await self._extract_images_from_message(
                msg, stats_list=optimization_stats
            )
            if parts:
                history_images.extend(parts)
        return history_images

    def _deduplicate_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.valves.IMAGE_DEDUP_HISTORY:
            return images
        seen: set[str] = set()
        result: List[Dict[str, Any]] = []
        for part in images:
            try:
                data = part["inline_data"]["data"]
                # Hash full base64 payload for stronger dedup reliability
                h = hashlib.sha256(data.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
            except Exception as e:
                # Skip images with malformed or missing data, but log for debugging.
                self.log.debug(f"Skipping image in deduplication due to error: {e}")
            result.append(part)
        return result

    def _combine_system_prompts(
        self, user_system_prompt: Optional[str]
    ) -> Optional[str]:
        """Combine default system prompt with user-defined system prompt.

        If DEFAULT_SYSTEM_PROMPT is set and user_system_prompt exists,
        the default is prepended to the user's prompt.
        If only DEFAULT_SYSTEM_PROMPT is set, it is used as the system prompt.
        If only user_system_prompt is set, it is used as-is.

        Args:
            user_system_prompt: The user-defined system prompt from messages (may be None)

        Returns:
            Combined system prompt or None if neither is set
        """
        default_prompt = self.valves.DEFAULT_SYSTEM_PROMPT.strip()
        user_prompt = user_system_prompt.strip() if user_system_prompt else ""

        if default_prompt and user_prompt:
            combined = f"{default_prompt}\n\n{user_prompt}"
            self.log.debug(
                f"Combined system prompts: default ({len(default_prompt)} chars) + "
                f"user ({len(user_prompt)} chars) = {len(combined)} chars"
            )
            return combined
        elif default_prompt:
            self.log.debug(f"Using default system prompt ({len(default_prompt)} chars)")
            return default_prompt
        elif user_prompt:
            return user_prompt
        return None

    def _apply_order_and_limit(
        self,
        history: List[Dict[str, Any]],
        current: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        """Combine history & current image parts honoring order & global limit.

        Returns:
            (combined_parts, reused_flags) where reused_flags[i] == True indicates
            the image originated from history, False if from current message.
        """
        history_first = self.valves.IMAGE_HISTORY_FIRST
        limit = max(1, self.valves.IMAGE_HISTORY_MAX_REFERENCES)
        combined: List[Dict[str, Any]] = []
        reused_flags: List[bool] = []

        def append(parts: List[Dict[str, Any]], reused: bool):
            for p in parts:
                if len(combined) >= limit:
                    break
                combined.append(p)
                reused_flags.append(reused)

        if history_first:
            append(history, True)
            append(current, False)
        else:
            append(current, False)
            append(history, True)
        return combined, reused_flags

    async def _emit_image_stats(
        self,
        ordered_stats: List[Dict[str, Any]],
        reused_flags: List[bool],
        total_limit: int,
        __event_emitter__: Callable,
    ) -> None:
        """Emit per-image optimization stats aligned with final combined order.

        ordered_stats: stats list in the exact order images will be sent (same length as combined image list)
        reused_flags: parallel list indicating whether image originated from history
        """
        if not ordered_stats:
            return
        for idx, stat in enumerate(ordered_stats, start=1):
            reused = reused_flags[idx - 1] if idx - 1 < len(reused_flags) else False
            stat_copy = dict(stat) if stat else {}
            stat_copy.update({"index": idx, "reused": reused})
            if stat and stat.get("original_size_mb") is not None:
                desc = f"Image {idx}: {stat['original_size_mb']:.2f}MB -> {stat['final_size_mb']:.2f}MB"
                if stat.get("quality") is not None:
                    desc += f" (Q{stat['quality']})"
            else:
                desc = f"Image {idx}: (no metrics)"
            reasons = stat.get("reasons") if stat else None
            if reasons:
                desc += " | " + ", ".join(reasons[:3])
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_optimization",
                        "description": desc,
                        "index": idx,
                        "done": False,
                        "details": stat_copy,
                    },
                }
            )
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "action": "image_optimization",
                    "description": f"{len(ordered_stats)} image(s) processed (limit {total_limit}).",
                    "done": True,
                },
            }
        )

    def __init__(self):
        """Initializes the Pipe instance and configures the genai library."""
        self.valves = self.Valves()
        self.name: str = "Google Gemini: "

        # Setup logging
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

        # Model cache
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_signature: str = ""

        # Multi-key round-robin state
        self._api_key_lock = threading.Lock()
        self._api_key_index: int = 0

    def _get_api_keys(self) -> List[str]:
        """Get configured API keys (supports comma-separated list)."""
        decrypted = EncryptedStr.decrypt(self.valves.GOOGLE_API_KEY)
        return [k.strip() for k in decrypted.split(",") if k and k.strip()]

    def _get_next_api_key(self) -> str:
        """Return next API key using round-robin selection."""
        keys = self._get_api_keys()
        if not keys:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please provide one or more API keys (comma-separated)."
            )

        with self._api_key_lock:
            key = keys[self._api_key_index % len(keys)]
            self._api_key_index = (self._api_key_index + 1) % len(keys)

        return key

    def _get_client(self) -> genai.Client:
        """
        Validates API credentials and returns a genai.Client instance.
        """
        self._validate_api_key()

        api_key = self._get_next_api_key()
        self.log.debug("Initializing Google Generative AI client with API Key")
        headers = {}
        if (
            self.valves.ENABLE_FORWARD_USER_INFO_HEADERS
            and hasattr(self, "user")
            and self.user
        ):

            def sanitize_header_value(value: Any, max_length: int = 255) -> str:
                if value is None:
                    return ""
                # Convert to string and remove all control characters
                sanitized = re.sub(r"[\x00-\x1F\x7F]", "", str(value))
                sanitized = sanitized.strip()
                return (
                    sanitized[:max_length]
                    if len(sanitized) > max_length
                    else sanitized
                )

            user_attrs = {
                "X-OpenWebUI-User-Name": sanitize_header_value(
                    getattr(self.user, "name", None)
                ),
                "X-OpenWebUI-User-Id": sanitize_header_value(
                    getattr(self.user, "id", None)
                ),
                "X-OpenWebUI-User-Email": sanitize_header_value(
                    getattr(self.user, "email", None)
                ),
                "X-OpenWebUI-User-Role": sanitize_header_value(
                    getattr(self.user, "role", None)
                ),
            }
            headers = {k: v for k, v in user_attrs.items() if v not in (None, "")}
        options = types.HttpOptions(
            api_version=self.valves.API_VERSION,
            base_url=self.valves.BASE_URL,
            headers=headers,
        )
        return genai.Client(
            api_key=api_key,
            http_options=options,
        )

    def _validate_api_key(self) -> None:
        """
        Validates that the necessary Google API credentials are set.

        Raises:
            ValueError: If the required credentials are not set.
        """
        api_keys = self._get_api_keys()
        if not api_keys:
            self.log.error("GOOGLE_API_KEY is not set.")
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please provide one or more API keys (comma-separated) in environment variables or valves."
            )
        self.log.debug(
            f"Using Google Generative AI API with {len(api_keys)} configured API key(s)."
        )

    def strip_prefix(self, model_name: str) -> str:
        """
        Extract the model identifier using regex, handling various naming conventions.
        e.g., "google_gemini_pipeline.gemini-2.5-flash-preview-04-17" -> "gemini-2.5-flash-preview-04-17"
        e.g., "models/gemini-1.5-flash-001" -> "gemini-1.5-flash-001"
        e.g., "publishers/google/models/gemini-1.5-pro" -> "gemini-1.5-pro"
        """
        model_name = (model_name or "").strip()
        if not model_name:
            return model_name

        # Handle path-style model names returned by SDK or API wrappers
        if "/" in model_name:
            return model_name.rsplit("/", 1)[-1]

        # Handle namespaced pipeline ids like "xxx.gemini-2.5-flash"
        # but keep native Gemini IDs such as "gemini-2.5-flash" intact.
        if "." in model_name and not model_name.startswith("gemini-"):
            return model_name.split(".", 1)[-1]

        return model_name

    def _parse_manual_model_entries(self, raw_models: str) -> List[Dict[str, str]]:
        """Parse manual model definitions from a comma-separated string.

        Supported formats per item:
        - model_id
        - model_id:display_name
        """
        parsed_models: List[Dict[str, str]] = []
        for item in re.findall(r"[^,]+", raw_models or ""):
            token = item.strip()
            if not token:
                continue

            if ":" in token:
                model_id_raw, display_name_raw = token.split(":", 1)
                display_name = display_name_raw.strip()
            else:
                model_id_raw = token
                display_name = ""

            model_id = self.strip_prefix(model_id_raw.strip())
            if not model_id:
                continue

            parsed_models.append(
                {
                    "id": model_id,
                    "name": display_name or model_id,
                }
            )

        return parsed_models

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """
        Retrieve available Google models suitable for content generation.
        Uses manual model configuration and cache.

        Args:
            force_refresh: Whether to force refreshing the model cache

        Returns:
            List of dictionaries containing model id and name.
        """
        # Check cache first
        cache_signature = "|".join(
            [
                self.valves.MODEL_LIST,
                str(self.valves.ENABLE_GEMINI_25_FLASH_SEARCH_MODEL),
            ]
        )
        if (
            not force_refresh
            and self._model_cache is not None
            and self._model_cache_signature == cache_signature
        ):
            self.log.debug("Using cached model list")
            return self._model_cache

        try:
            self.log.debug("Building models from manual configuration")

            model_map = {
                model["id"]: model
                for model in self._parse_manual_model_entries(self.valves.MODEL_LIST)
            }

            # Keep only gemini-* models for safety
            filtered_models = {
                k: v for k, v in model_map.items() if k.startswith("gemini-")
            }
            self.log.debug(f"After gemini-prefix filter: {len(filtered_models)} models")

            if self.valves.ENABLE_GEMINI_25_FLASH_SEARCH_MODEL:
                filtered_models["gemini-2.5-flash-search"] = {
                    "id": "gemini-2.5-flash-search",
                    "name": "Gemini 2.5 Flash (Search) 🔎",
                }

            # Update cache
            self._model_cache = list(filtered_models.values())
            self._model_cache_signature = cache_signature
            self.log.debug(f"Found {len(self._model_cache)} Gemini models")
            return self._model_cache

        except Exception as e:
            self.log.exception(f"Could not build models from manual config: {str(e)}")
            # Return a specific error entry for the UI
            return [{"id": "error", "name": f"Could not build model list: {str(e)}"}]

    def _check_thinking_support(self, model_id: str) -> bool:
        """
        Check if a model supports the thinking feature.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports thinking, False otherwise
        """
        # By default, assume models support thinking
        return True

    def _check_thinking_level_support(self, model_id: str) -> bool:
        """
        Check if a model supports the thinking_level parameter.

        Gemini 3 models support thinking_level and should NOT use thinking_budget.
        Other models (like Gemini 2.5) use thinking_budget instead.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports thinking_level, False otherwise
        """
        # Gemini 3 models support thinking_level (not thinking_budget)
        gemini_3_patterns = [
            "gemini-3",
        ]

        model_lower = model_id.lower()
        for pattern in gemini_3_patterns:
            if pattern in model_lower:
                return True

        return False

    def _validate_thinking_level(self, level: str) -> Optional[str]:
        """
        Validate and normalize the thinking level value.

        Args:
            level: The thinking level string to validate

        Returns:
            Normalized level string ('minimal', 'low', 'medium', 'high') or None if invalid/empty
        """
        if not level:
            return None

        normalized = level.strip().lower()
        valid_levels = ["minimal", "low", "medium", "high"]

        if normalized in valid_levels:
            return normalized

        self.log.warning(
            f"Invalid thinking level '{level}'. Valid values are: {', '.join(valid_levels)}. "
            "Falling back to model default."
        )
        return None

    def _validate_thinking_budget(self, budget: int) -> int:
        """
        Validate and normalize the thinking budget value.

        Args:
            budget: The thinking budget integer to validate

        Returns:
            Validated budget: -1 for dynamic, 0 to disable, or 1-32768 for fixed limit
        """
        # -1 means dynamic thinking (let the model decide)
        if budget == -1:
            return -1

        # 0 means disable thinking
        if budget == 0:
            return 0

        # Validate positive range (1-32768)
        if budget > 0:
            if budget > 32768:
                self.log.warning(
                    f"Thinking budget {budget} exceeds maximum of 32768. Clamping to 32768."
                )
                return 32768
            return budget

        # Negative values (except -1) are invalid, treat as -1 (dynamic)
        self.log.warning(
            f"Invalid thinking budget {budget}. Only -1 (dynamic), 0 (disabled), or 1-32768 are valid. "
            "Falling back to dynamic thinking."
        )
        return -1

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available Google Gemini models for the UI.

        Returns:
            List of dictionaries containing model id and name.
        """
        try:
            self.name = "Google Gemini: "
            return self.get_google_models()
        except ValueError as e:
            # Handle the case where API key is missing during pipe listing
            self.log.error(f"Error during pipes listing (validation): {e}")
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            # Handle other potential errors during model fetching
            self.log.exception(
                f"An unexpected error occurred during pipes listing: {str(e)}"
            )
            return [{"id": "error", "name": f"An unexpected error occurred: {str(e)}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        """
        Prepare and validate the model ID for use with the API.

        Args:
            model_id: The original model ID from the user

        Returns:
            Properly formatted model ID

        Raises:
            ValueError: If the model ID is invalid or unsupported
        """
        original_model_id = model_id
        model_id = self.strip_prefix(model_id)

        # If the model ID doesn't look like a Gemini model, try to find it by name
        if not model_id.startswith("gemini-"):
            models_list = self.get_google_models()
            found_model = next(
                (m["id"] for m in models_list if m["name"] == original_model_id), None
            )
            if found_model and found_model.startswith("gemini-"):
                model_id = found_model
                self.log.debug(
                    f"Mapped model name '{original_model_id}' to model ID '{model_id}'"
                )
            else:
                # If we still don't have a valid ID, raise an error
                if not model_id.startswith("gemini-"):
                    self.log.error(
                        f"Invalid or unsupported model ID: '{original_model_id}'"
                    )
                    raise ValueError(
                        f"Invalid or unsupported Google model ID or name: '{original_model_id}'"
                    )

        return model_id

    def _prepare_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Prepare messages content for the API and extract system message if present.

        Args:
            messages: List of message objects from the request

        Returns:
            Tuple of (prepared content list, system message string or None)
        """
        # Extract user-defined system message
        user_system_message = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        # Combine with default system prompt if configured
        system_message = self._combine_system_prompts(user_system_message)

        # Prepare contents for the API
        contents = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue  # Skip system messages, handled separately

            content = message.get("content", "")

            # Avoid feeding prior thought-details blocks back into context.
            # This prevents follow-up turns from being polluted by rendered thinking text.
            if role == "assistant":
                if isinstance(content, str):
                    content = self._strip_thought_details(content)
                    if not content.strip():
                        continue
                elif isinstance(content, list):
                    sanitized_items: List[Dict[str, Any]] = []
                    for item in content:
                        if not isinstance(item, dict):
                            sanitized_items.append(item)
                            continue
                        if item.get("type") != "text":
                            sanitized_items.append(item)
                            continue
                        text_value = item.get("text", "")
                        cleaned_text = self._strip_thought_details(
                            text_value if isinstance(text_value, str) else ""
                        )
                        if cleaned_text.strip():
                            sanitized_items.append({**item, "text": cleaned_text})
                    content = sanitized_items

            parts = []

            # Handle different content types
            if isinstance(content, list):  # Multimodal content
                parts.extend(self._process_multimodal_content(content))
            elif isinstance(content, str):  # Plain text content
                parts.append({"text": content})
            else:
                self.log.warning(f"Unsupported message content type: {type(content)}")
                continue  # Skip unsupported content

            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            api_role = "model" if role == "assistant" else "user"
            if parts:  # Only add if there are parts
                contents.append({"role": api_role, "parts": parts})

        return contents, system_message

    def _strip_thought_details(self, text: str) -> str:
        """Remove rendered thought blocks from assistant history content."""
        if not text:
            return text

        thought_details_pattern = re.compile(
            r"<details(?:\s+[^>]*)?>\s*<summary>\s*(?:思考过程|thinking\s*process)[^<]*</summary>[\s\S]*?</details>\s*",
            re.IGNORECASE,
        )
        cleaned = thought_details_pattern.sub("", text)
        return cleaned.strip()

    def _process_multimodal_content(
        self, content_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multimodal content (text and images).

        Args:
            content_list: List of content items

        Returns:
            List of processed parts for the Gemini API
        """
        parts = []

        for item in content_list:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")

                if image_url.startswith("data:image"):
                    # Handle base64 encoded image data with optimization
                    try:
                        # Optimize the image before processing
                        optimized_image = self._optimize_image_for_api(image_url)
                        header, encoded = optimized_image.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        # Basic validation for image types
                        if mime_type not in [
                            "image/jpeg",
                            "image/png",
                            "image/webp",
                            "image/heic",
                            "image/heif",
                        ]:
                            self.log.warning(
                                f"Unsupported image mime type: {mime_type}"
                            )
                            parts.append(
                                {"text": f"[Image type {mime_type} not supported]"}
                            )
                            continue

                        # Check if the encoded data is too large
                        if len(encoded) > 15 * 1024 * 1024:  # 15MB limit for base64
                            self.log.warning(
                                f"Image data too large: {len(encoded)} characters"
                            )
                            parts.append(
                                {
                                    "text": "[Image too large for processing - please use a smaller image]"
                                }
                            )
                            continue

                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded,
                                }
                            }
                        )
                    except Exception as img_ex:
                        self.log.exception(f"Could not parse image data URL: {img_ex}")
                        parts.append({"text": "[Image data could not be processed]"})
                else:
                    # Gemini API doesn't directly support image URLs
                    self.log.warning(f"Direct image URLs not supported: {image_url}")
                    parts.append({"text": f"[Image URL not processed: {image_url}]"})

        return parts

    # _find_image removed (was single-image oriented and is superseded by multi-image logic)

    async def _extract_images_from_message(
        self,
        message: Dict[str, Any],
        *,
        stats_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract prompt text and ALL images from a single user message.

        This replaces the previous single-image _find_image logic for image-capable
        models so that multi-image prompts are respected.

        Returns:
            (prompt_text, image_parts)
                prompt_text: concatenated text content (may be empty)
                image_parts: list of {"inline_data": {mime_type, data}} dicts
        """
        content = message.get("content", "")
        text_segments: List[str] = []
        image_parts: List[Dict[str, Any]] = []

        # Helper to process a data URL or fetched file and append inline_data
        def _add_image(data_url: str):
            try:
                optimized = self._optimize_image_for_api(data_url, stats_list)
                header, b64 = optimized.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0]
                image_parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            except Exception as e:  # pragma: no cover - defensive
                self.log.warning(f"Skipping image (parse failure): {e}")

        # Regex to extract markdown image references
        md_pattern = re.compile(
            r"!\[[^\]]*\]\((data:image[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)"
        )

        # Structured multimodal array
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    text_segments.append(txt)
                    # Also parse any markdown images embedded in the text
                    for match in md_pattern.finditer(txt):
                        url = match.group(1)
                        if url.startswith("data:"):
                            _add_image(url)
                        else:
                            b64 = await self._fetch_file_as_base64(url)
                            if b64:
                                _add_image(b64)
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        _add_image(url)
                    elif "/files/" in url or "/api/v1/files/" in url:
                        b64 = await self._fetch_file_as_base64(url)
                        if b64:
                            _add_image(b64)
        # Plain string message (may include markdown images)
        elif isinstance(content, str):
            text_segments.append(content)
            for match in md_pattern.finditer(content):
                url = match.group(1)
                if url.startswith("data:"):
                    _add_image(url)
                else:
                    b64 = await self._fetch_file_as_base64(url)
                    if b64:
                        _add_image(b64)
        else:
            self.log.debug(
                f"Unsupported content type for image extraction: {type(content)}"
            )

        prompt_text = " ".join(s.strip() for s in text_segments if s.strip())
        return prompt_text, image_parts

    def _optimize_image_for_api(
        self, image_data: str, stats_list: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Optimize image data for Gemini API using configurable parameters.

        Returns:
            Optimized base64 data URL
        """
        # Check if optimization is enabled
        if not self.valves.IMAGE_ENABLE_OPTIMIZATION:
            self.log.debug("Image optimization disabled via configuration")
            return image_data

        max_size_mb = self.valves.IMAGE_MAX_SIZE_MB
        max_dimension = self.valves.IMAGE_MAX_DIMENSION
        base_quality = self.valves.IMAGE_COMPRESSION_QUALITY
        png_threshold = self.valves.IMAGE_PNG_COMPRESSION_THRESHOLD_MB

        self.log.debug(
            f"Image optimization config: max_size={max_size_mb}MB, max_dim={max_dimension}px, quality={base_quality}, png_threshold={png_threshold}MB"
        )
        try:
            # Parse the data URL
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
            else:
                encoded = image_data
                mime_type = "image/png"

            # Decode and analyze the image
            image_bytes = base64.b64decode(encoded)
            original_size_mb = len(image_bytes) / (1024 * 1024)
            base64_size_mb = len(encoded) / (1024 * 1024)

            self.log.debug(
                f"Original image: {original_size_mb:.2f} MB (decoded), {base64_size_mb:.2f} MB (base64), type: {mime_type}"
            )

            # Determine optimization strategy
            reasons: List[str] = []
            if original_size_mb > max_size_mb:
                reasons.append(f"size > {max_size_mb} MB")
            if base64_size_mb > max_size_mb * 1.4:
                reasons.append("base64 overhead")
            if mime_type == "image/png" and original_size_mb > png_threshold:
                reasons.append(f"PNG > {png_threshold}MB")

            # Always check dimensions
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                resized_flag = False
                if width > max_dimension or height > max_dimension:
                    reasons.append(f"dimensions > {max_dimension}px")

                # Early exit: no optimization triggers -> keep original, record stats
                if not reasons:
                    if stats_list is not None:
                        stats_list.append(
                            {
                                "original_size_mb": round(original_size_mb, 4),
                                "final_size_mb": round(original_size_mb, 4),
                                "quality": None,
                                "format": mime_type.split("/")[-1].upper(),
                                "resized": False,
                                "reasons": ["no_optimization_needed"],
                                "final_hash": hashlib.sha256(
                                    encoded.encode()
                                ).hexdigest(),
                            }
                        )
                    self.log.debug(
                        "Skipping optimization: image already within thresholds"
                    )
                    return image_data

                self.log.debug(f"Optimization triggers: {', '.join(reasons)}")

                # Convert to RGB for JPEG compression
                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(
                        img,
                        mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None,
                    )
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if needed
                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    self.log.debug(
                        f"Resizing from {width}x{height} to {new_size[0]}x{new_size[1]}"
                    )
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_flag = True

                # Determine quality levels based on original size and user configuration
                if original_size_mb > 5.0:
                    quality_levels = [
                        base_quality,
                        base_quality - 10,
                        base_quality - 20,
                        base_quality - 30,
                        base_quality - 40,
                        max(base_quality - 50, 25),
                    ]
                elif original_size_mb > 2.0:
                    quality_levels = [
                        base_quality,
                        base_quality - 5,
                        base_quality - 15,
                        base_quality - 25,
                        max(base_quality - 35, 35),
                    ]
                else:
                    quality_levels = [
                        min(base_quality + 5, 95),
                        base_quality,
                        base_quality - 10,
                        max(base_quality - 20, 50),
                    ]

                # Ensure quality levels are within valid range (1-100)
                quality_levels = [max(1, min(100, q)) for q in quality_levels]

                # Try compression levels
                for quality in quality_levels:
                    output_buffer = io.BytesIO()
                    format_type = (
                        "JPEG"
                        if original_size_mb > png_threshold or "jpeg" in mime_type
                        else "PNG"
                    )
                    output_mime = f"image/{format_type.lower()}"

                    img.save(
                        output_buffer,
                        format=format_type,
                        quality=quality,
                        optimize=True,
                    )
                    output_bytes = output_buffer.getvalue()
                    output_size_mb = len(output_bytes) / (1024 * 1024)

                    if output_size_mb <= max_size_mb:
                        optimized_b64 = base64.b64encode(output_bytes).decode("utf-8")
                        self.log.debug(
                            f"Optimized: {original_size_mb:.2f} MB → {output_size_mb:.2f} MB (Q{quality})"
                        )
                        if stats_list is not None:
                            stats_list.append(
                                {
                                    "original_size_mb": round(original_size_mb, 4),
                                    "final_size_mb": round(output_size_mb, 4),
                                    "quality": quality,
                                    "format": format_type,
                                    "resized": resized_flag,
                                    "reasons": reasons,
                                    "final_hash": hashlib.sha256(
                                        optimized_b64.encode()
                                    ).hexdigest(),
                                }
                            )
                        return f"data:{output_mime};base64,{optimized_b64}"

                # Fallback: minimum quality
                output_buffer = io.BytesIO()
                img.save(output_buffer, format="JPEG", quality=15, optimize=True)
                output_bytes = output_buffer.getvalue()
                output_size_mb = len(output_bytes) / (1024 * 1024)
                optimized_b64 = base64.b64encode(output_bytes).decode("utf-8")

                self.log.warning(
                    f"Aggressive optimization: {output_size_mb:.2f} MB (Q15)"
                )
                if stats_list is not None:
                    stats_list.append(
                        {
                            "original_size_mb": round(original_size_mb, 4),
                            "final_size_mb": round(output_size_mb, 4),
                            "quality": 15,
                            "format": "JPEG",
                            "resized": resized_flag,
                            "reasons": reasons + ["fallback_min_quality"],
                            "final_hash": hashlib.sha256(
                                optimized_b64.encode()
                            ).hexdigest(),
                        }
                    )
                return f"data:image/jpeg;base64,{optimized_b64}"

        except Exception as e:
            self.log.error(f"Image optimization failed: {e}")
            # Return original or safe fallback
            if image_data.startswith("data:"):
                if stats_list is not None:
                    stats_list.append(
                        {
                            "original_size_mb": None,
                            "final_size_mb": None,
                            "quality": None,
                            "format": None,
                            "resized": False,
                            "reasons": ["optimization_failed"],
                            "final_hash": (
                                hashlib.sha256(encoded.encode()).hexdigest()
                                if "encoded" in locals()
                                else None
                            ),
                        }
                    )
                return image_data
            return f"data:image/jpeg;base64,{encoded if 'encoded' in locals() else image_data}"

    async def _fetch_file_as_base64(self, file_url: str) -> Optional[str]:
        """
        Fetch a file from Open WebUI's file system and convert to base64.

        Args:
            file_url: File URL from Open WebUI

        Returns:
            Base64 encoded file data or None if file not found
        """
        try:
            if "/api/v1/files/" in file_url:
                fid = file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
            else:
                fid = file_url.split("/files/")[-1].split("/")[0].split("?")[0]

            from pathlib import Path
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage

            file_obj = Files.get_file_by_id(fid)
            if file_obj and file_obj.path:
                file_path = Storage.get_file(file_obj.path)
                file_path = Path(file_path)
                if file_path.is_file():
                    async with aiofiles.open(file_path, "rb") as fp:
                        raw = await fp.read()
                    enc = base64.b64encode(raw).decode()
                    mime = file_obj.meta.get("content_type", "image/png")
                    return f"data:{mime};base64,{enc}"
        except Exception as e:
            self.log.warning(f"Could not fetch file {file_url}: {e}")
        return None

    def _get_user_valve_value(
        self, __user__: Optional[dict], valve_name: str
    ) -> Optional[str]:
        """Get a user valve value, returning None if not set or set to 'default'"""
        if __user__ and "valves" in __user__:
            value = getattr(__user__["valves"], valve_name, None)
            if value and value != "default":
                return value
        return None

    def _resolve_include_thoughts(self, body: Dict[str, Any]) -> bool:
        """Resolve include_thoughts for current request with valve fallback."""
        include_thoughts_raw = body.get("include_thoughts", None)
        if include_thoughts_raw is None:
            include_thoughts = True
        elif isinstance(include_thoughts_raw, str):
            include_thoughts = include_thoughts_raw.strip().lower() == "true"
        else:
            include_thoughts = bool(include_thoughts_raw)

        if not self.valves.INCLUDE_THOUGHTS:
            include_thoughts = False
            self.log.debug("Thoughts output disabled via GOOGLE_INCLUDE_THOUGHTS")

        return include_thoughts

    def _configure_generation(
        self,
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        __tools__: dict[str, Any] | None = None,
        __user__: Optional[dict] = None,
        model_id: str = "",
        force_google_search_tool: bool = False,
    ) -> types.GenerateContentConfig:
        """
        Configure generation parameters and safety settings.

        Args:
            body: The request body containing generation parameters
            system_instruction: Optional system instruction string
            model_id: The model ID being used (for feature support checks)

        Returns:
            types.GenerateContentConfig
        """
        gen_config_params = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop_sequences": body.get("stop") or None,
            "system_instruction": system_instruction,
        }

        # Configure Gemini thinking/reasoning for models that support it
        # This is independent of include_thoughts - thinking config controls HOW the model reasons,
        # while include_thoughts controls whether the reasoning is shown in the output
        if self._check_thinking_support(model_id):
            try:
                thinking_config_params: Dict[str, Any] = {}

                # Determine include_thoughts setting
                include_thoughts = self._resolve_include_thoughts(body)
                thinking_config_params["include_thoughts"] = include_thoughts

                # Check if model supports thinking_level (Gemini 3 models)
                if self._check_thinking_level_support(model_id):
                    # For Gemini 3 models, use thinking_level (not thinking_budget)
                    # Per-chat reasoning_effort overrides environment-level THINKING_LEVEL
                    reasoning_effort = body.get("reasoning_effort")
                    validated_level = None
                    source = None

                    if reasoning_effort:
                        validated_level = self._validate_thinking_level(
                            reasoning_effort
                        )
                        if validated_level:
                            source = "per-chat reasoning_effort"
                        else:
                            self.log.debug(
                                f"Invalid reasoning_effort '{reasoning_effort}', falling back to THINKING_LEVEL"
                            )

                    # Fall back to environment-level THINKING_LEVEL if no valid reasoning_effort
                    if not validated_level:
                        validated_level = self._validate_thinking_level(
                            self.valves.THINKING_LEVEL
                        )
                        if validated_level:
                            source = "THINKING_LEVEL"

                    if validated_level:
                        thinking_config_params["thinking_level"] = validated_level
                        self.log.debug(
                            f"Using thinking_level='{validated_level}' from {source} for model {model_id}"
                        )
                    else:
                        self.log.debug(
                            f"Using default thinking level for model {model_id}"
                        )
                else:
                    # For non-Gemini 3 models (e.g., Gemini 2.5), use thinking_budget
                    # Body-level thinking_budget overrides environment-level THINKING_BUDGET
                    body_thinking_budget = body.get("thinking_budget")
                    validated_budget = None
                    source = None

                    if body_thinking_budget is not None:
                        validated_budget = self._validate_thinking_budget(
                            body_thinking_budget
                        )
                        if validated_budget is not None:
                            source = "body thinking_budget"
                        else:
                            self.log.debug(
                                f"Invalid body thinking_budget '{body_thinking_budget}', falling back to THINKING_BUDGET"
                            )

                    # Fall back to environment-level THINKING_BUDGET
                    if validated_budget is None:
                        validated_budget = self._validate_thinking_budget(
                            self.valves.THINKING_BUDGET
                        )
                        if validated_budget is not None:
                            source = "THINKING_BUDGET"

                    if validated_budget == 0:
                        # Disable thinking if budget is 0
                        thinking_config_params["thinking_budget"] = 0
                        self.log.debug(
                            f"Thinking disabled via thinking_budget=0 from {source} for model {model_id}"
                        )
                    elif validated_budget is not None and validated_budget > 0:
                        thinking_config_params["thinking_budget"] = validated_budget
                        self.log.debug(
                            f"Using thinking_budget={validated_budget} from {source} for model {model_id}"
                        )
                    else:
                        # -1 or None means dynamic thinking
                        thinking_config_params["thinking_budget"] = -1
                        self.log.debug(
                            f"Using dynamic thinking (model decides) for model {model_id}"
                        )

                gen_config_params["thinking_config"] = types.ThinkingConfig(
                    **thinking_config_params
                )
            except (AttributeError, TypeError) as e:
                # Fall back if SDK/model does not support ThinkingConfig
                self.log.debug(f"ThinkingConfig not supported: {e}")
            except Exception as e:
                # Log unexpected errors but continue without thinking config
                self.log.warning(f"Unexpected error configuring ThinkingConfig: {e}")

        # Configure safety settings
        if self.valves.USE_PERMISSIVE_SAFETY:
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
                ),
            ]
            gen_config_params |= {"safety_settings": safety_settings}

        # Add various tools to Gemini as required
        features = __metadata__.get("features", {})
        params = __metadata__.get("params", {})
        tools = []
        google_search_added = False

        if features.get("google_search_tool", False):
            if model_id.startswith("gemini-2.5-flash"):
                self.log.debug(
                    "Enabling built-in Google Search tool for gemini-2.5-flash via chat switch"
                )
                tools.append(types.Tool(google_search=types.GoogleSearch()))
                google_search_added = True
            else:
                self.log.debug("Enabling Google search grounding")
                tools.append(types.Tool(google_search=types.GoogleSearch()))
                google_search_added = True
                self.log.debug("Enabling URL context grounding")
                tools.append(types.Tool(url_context=types.UrlContext()))

        if force_google_search_tool and not google_search_added:
            self.log.debug("Enabling built-in Google Search tool via search model")
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        if __tools__ is not None and params.get("function_calling") == "native":
            for name, tool_def in __tools__.items():
                if not name.startswith("_"):
                    tool = tool_def["callable"]
                    self.log.debug(
                        f"Adding tool '{name}' with signature {tool.__signature__}"
                    )
                    tools.append(tool)

        if tools:
            gen_config_params["tools"] = tools

        # Filter out None values for generation config
        filtered_params = {k: v for k, v in gen_config_params.items() if v is not None}
        return types.GenerateContentConfig(**filtered_params)

    @staticmethod
    def _format_grounding_chunks_as_sources(
        grounding_chunks: list[types.GroundingChunk],
    ):
        formatted_sources = []
        for chunk in grounding_chunks:
            if hasattr(chunk, "retrieved_context") and chunk.retrieved_context:
                context = chunk.retrieved_context
                formatted_sources.append(
                    {
                        "source": {
                            "name": getattr(context, "title", None) or "Document",
                            "type": "document",
                            "uri": getattr(context, "uri", None),
                        },
                        "document": [getattr(context, "chunk_text", None) or ""],
                        "metadata": [
                            {"source": getattr(context, "title", None) or "Document"}
                        ],
                    }
                )
            elif hasattr(chunk, "web") and chunk.web:
                context = chunk.web
                uri = context.uri
                title = context.title or "Source"

                formatted_sources.append(
                    {
                        "source": {
                            "name": title,
                            "type": "web_search_results",
                            "url": uri,
                        },
                        "document": ["Click the link to view the content."],
                        "metadata": [{"source": title}],
                    }
                )
        return formatted_sources

    async def _process_grounding_metadata(
        self,
        grounding_metadata_list: List[types.GroundingMetadata],
        text: str,
        __event_emitter__: Callable,
    ):
        """Process and emit grounding metadata events."""
        grounding_chunks = []
        web_search_queries = []
        grounding_supports = []

        for metadata in grounding_metadata_list:
            if metadata.grounding_chunks:
                grounding_chunks.extend(metadata.grounding_chunks)
            if metadata.web_search_queries:
                web_search_queries.extend(metadata.web_search_queries)
            if metadata.grounding_supports:
                grounding_supports.extend(metadata.grounding_supports)

        # Add sources to the response
        if grounding_chunks:
            sources = self._format_grounding_chunks_as_sources(grounding_chunks)
            await __event_emitter__(
                {"type": "chat:completion", "data": {"sources": sources}}
            )

        # Add status specifying google queries used for grounding
        if web_search_queries:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "This response was grounded with Google Search",
                        "urls": [
                            f"https://www.google.com/search?q={query}"
                            for query in web_search_queries
                        ],
                    },
                }
            )

        # Add citations in the text body
        replaced_text: Optional[str] = None
        if grounding_supports:
            # Citation indexes are in bytes
            ENCODING = "utf-8"
            text_bytes = text.encode(ENCODING)
            last_byte_index = 0
            cited_chunks = []

            for support in grounding_supports:
                cited_chunks.append(
                    text_bytes[last_byte_index : support.segment.end_index].decode(
                        ENCODING
                    )
                )

                # Generate and append citations (e.g., "[1][2]")
                footnotes = "".join(
                    [f"[{i + 1}]" for i in support.grounding_chunk_indices]
                )
                cited_chunks.append(f" {footnotes}")

                # Update index for the next segment
                last_byte_index = support.segment.end_index

            # Append any remaining text after the last citation
            if last_byte_index < len(text_bytes):
                cited_chunks.append(text_bytes[last_byte_index:].decode(ENCODING))

            replaced_text = "".join(cited_chunks)

        return replaced_text if replaced_text is not None else text

    async def _handle_streaming_response(
        self,
        response_iterator: Any,
        __event_emitter__: Callable,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
        retry_stream_factory: Optional[Callable] = None,
        max_empty_retries: Optional[int] = None,
        max_stream_error_retries: Optional[int] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Handle streaming response from Gemini API.

        Args:
            response_iterator: Iterator from generate_content
            __event_emitter__: Event emitter for status updates

        Returns:
            Generator yielding text chunks
        """

        async def emit_chat_event(event_type: str, data: Dict[str, Any]) -> None:
            if not __event_emitter__:
                return
            try:
                await __event_emitter__({"type": event_type, "data": data})
            except Exception as emit_error:  # pragma: no cover - defensive
                self.log.warning(f"Failed to emit {event_type} event: {emit_error}")

        await emit_chat_event("chat:start", {"role": "assistant"})

        max_empty_retries = (
            self._get_empty_response_retry_count()
            if max_empty_retries is None
            else max(0, max_empty_retries)
        )
        empty_retry_count = 0
        max_stream_error_retries = (
            max(0, self.valves.RETRY_COUNT)
            if max_stream_error_retries is None
            else max(0, max_stream_error_retries)
        )
        stream_error_retry_count = 0

        while True:
            grounding_metadata_list = []
            answer_chunks: list[str] = []
            thought_chunks: list[str] = []
            thinking_started_at: Optional[float] = None
            stream_usage_metadata = None

            try:
                async for chunk in response_iterator:
                    # Capture usage metadata (final chunk has complete data)
                    if getattr(chunk, "usage_metadata", None):
                        stream_usage_metadata = chunk.usage_metadata

                    # Check for safety feedback or empty chunks
                    if not chunk.candidates:
                        # Check prompt feedback
                        if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                            block_reason = chunk.prompt_feedback.block_reason.name
                            message = f"[Blocked due to Prompt Safety: {block_reason}]"
                            await emit_chat_event(
                                "chat:finish",
                                {
                                    "role": "assistant",
                                    "content": message,
                                    "done": True,
                                    "error": True,
                                },
                            )
                            yield message
                        else:
                            message = "[Blocked by safety settings]"
                            await emit_chat_event(
                                "chat:finish",
                                {
                                    "role": "assistant",
                                    "content": message,
                                    "done": True,
                                    "error": True,
                                },
                            )
                            yield message
                        return  # Stop generation

                    if chunk.candidates[0].grounding_metadata:
                        grounding_metadata_list.append(
                            chunk.candidates[0].grounding_metadata
                        )
                    # Prefer fine-grained parts to split thoughts vs. normal text
                    parts = []
                    try:
                        parts = chunk.candidates[0].content.parts or []
                    except Exception as parts_error:
                        # Fallback: use aggregated text if parts aren't accessible
                        self.log.warning(f"Failed to access content parts: {parts_error}")
                        if hasattr(chunk, "text") and chunk.text:
                            answer_chunks.append(chunk.text)
                            await __event_emitter__(
                                {
                                    "type": "chat:message:delta",
                                    "data": {
                                        "role": "assistant",
                                        "content": chunk.text,
                                    },
                                }
                            )
                        continue

                    for part in parts:
                        try:
                            # Thought parts (internal reasoning)
                            if getattr(part, "thought", False) and getattr(
                                part, "text", None
                            ):
                                if thinking_started_at is None:
                                    thinking_started_at = time.time()
                                thought_chunks.append(part.text)
                                # Emit a live preview of what is currently being thought
                                preview = part.text.replace("\n", " ").strip()
                                # Remove markdown formatting and multiple spaces for cleaner preview
                                preview = re.sub(r'\*\*([^*]+)\*\*', r'\1', preview)  # Remove bold
                                preview = re.sub(r'\s+', ' ', preview)  # Normalize whitespace
                                MAX_PREVIEW = 120
                                if len(preview) > MAX_PREVIEW:
                                    preview = preview[:MAX_PREVIEW].rstrip() + "…"
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "action": "thinking",
                                            "description": f"正在思考：{preview}",
                                            "done": False,
                                            "hidden": False,
                                        },
                                    }
                                )

                            # Regular answer text
                            elif getattr(part, "text", None):
                                answer_chunks.append(part.text)
                                await __event_emitter__(
                                    {
                                        "type": "chat:message:delta",
                                        "data": {
                                            "role": "assistant",
                                            "content": part.text,
                                        },
                                    }
                                )
                        except Exception as part_error:
                            # Log part processing errors but continue with the stream
                            self.log.warning(f"Error processing content part: {part_error}")
                            continue

                # After processing all chunks, handle grounding data
                final_answer_text = "".join(answer_chunks)
                if grounding_metadata_list and __event_emitter__:
                    cited = await self._process_grounding_metadata(
                        grounding_metadata_list,
                        final_answer_text,
                        __event_emitter__,
                    )
                    final_answer_text = cited or final_answer_text

                final_content = final_answer_text
                details_block: Optional[str] = None

                if thought_chunks:
                    duration_s = int(
                        max(0, time.time() - (thinking_started_at or time.time()))
                    )
                    # Format each line with > for blockquote
                    thought_content = "".join(thought_chunks).strip()
                    quoted_lines = []
                    for line in thought_content.split("\n"):
                        quoted_lines.append(f"> {line}")
                    quoted_content = "\n".join(quoted_lines)

                    details_block = f"""<details>
<summary>思考过程 ({duration_s}s)</summary>

{quoted_content}

</details>"""
                    final_content = f"{details_block}\n\n{final_answer_text}"

                if not final_content:
                    final_content = ""

                if not final_answer_text.strip() and retry_stream_factory:
                    if empty_retry_count < max_empty_retries:
                        empty_retry_count += 1
                        self.log.warning(
                            f"Streaming returned empty content. Retrying ({empty_retry_count}/{max_empty_retries})"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "action": "retry",
                                        "description": f"模型返回空内容，正在自动重试（{empty_retry_count}/{max_empty_retries}）",
                                        "done": False,
                                    },
                                }
                            )
                        response_iterator = await self._retry_with_backoff(
                            retry_stream_factory
                        )
                        continue
                    elif max_empty_retries > 0:
                        # Max retries exceeded
                        if __event_emitter__ and empty_retry_count > 0:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "action": "retry",
                                        "description": f"无法获取有效回答：已达到最大重试次数（{max_empty_retries}）",
                                        "done": True,
                                    },
                                }
                            )
                        warning_msg = "⚠️ [模型多次尝试后未生成有效正文内容，可能由于安全策略拦截或当前无法作答]"
                        final_content = (final_content + f"\n\n{warning_msg}").strip()

                # Ensure downstream consumers (UI, TTS) receive the complete response once streaming ends.
                await emit_chat_event(
                    "replace", {"role": "assistant", "content": final_content}
                )
                await emit_chat_event(
                    "chat:message",
                    {"role": "assistant", "content": final_content, "done": True},
                )

                if thought_chunks:
                    # Clear the thinking status without a summary in the status emitter
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"action": "thinking", "done": True, "hidden": True},
                        }
                    )

                # Yield usage data as dict so the middleware can extract and save it to DB
                usage = self._build_usage_dict(stream_usage_metadata)
                if usage:
                    yield {"usage": usage}

                await emit_chat_event(
                    "chat:finish",
                    {"role": "assistant", "content": final_content, "done": True},
                )

                # Yield final content to ensure the async iterator completes properly.
                # This ensures the response is persisted even if the user navigates away.
                yield final_content
                return

            except Exception as e:
                is_expected_api_error = isinstance(
                    e, (ClientError, ServerError, APIError)
                ) and self._is_expected_api_error(e)
                if is_expected_api_error:
                    self.log.warning(
                        "Streaming got transient API error: "
                        f"{self._compact_error_message(e)}"
                    )
                else:
                    self.log.exception(f"Error during streaming: {e}")
                error_msg = str(e).lower()
                error_name = type(e).__name__

                recoverable_stream_error = (
                    "assert self._connector is not none" in error_msg
                    or "connector is not none" in error_msg
                    or error_name in ("ClientConnectorError", "ClientOSError", "ServerDisconnectedError", "ServerTimeoutError", "TimeoutError", "ClientPayloadError")
                    or isinstance(e, asyncio.TimeoutError)
                )
                if (
                    retry_stream_factory
                    and stream_error_retry_count < max_stream_error_retries
                    and (
                        self._is_retryable_api_error(e)
                        or recoverable_stream_error
                    )
                ):
                    stream_error_retry_count += 1
                    self.log.warning(
                        "Streaming failed with recoverable error. "
                        f"Retrying with next available key/source ({stream_error_retry_count}/{max_stream_error_retries}). "
                        f"{self._compact_error_message(e)}"
                    )
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "retry",
                                    "description": f"流式连接中断或额度受限，正在自动重试（{stream_error_retry_count}/{max_stream_error_retries}）",
                                    "done": False,
                                },
                            }
                        )
                    response_iterator = await self._retry_with_backoff(
                        retry_stream_factory
                    )
                    continue

                # Check if it's a chunk size error and provide specific guidance
                if "chunk too big" in error_msg or "chunk size" in error_msg:
                    message = "Error: Image too large for processing. Please try with a smaller image (max 15 MB recommended) or reduce image quality."
                elif self._is_service_unavailable_error(e):
                    message = self._service_unavailable_user_message()
                elif "quota" in error_msg or "rate limit" in error_msg:
                    message = "Error: API quota exceeded. Please try again later."
                else:
                    message = f"Error during streaming: {e}"
                await emit_chat_event(
                    "chat:finish",
                    {
                        "role": "assistant",
                        "content": message,
                        "done": True,
                        "error": True,
                    },
                )
                yield message
                return

    @staticmethod
    def _build_usage_dict(usage_metadata: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from Gemini usage_metadata into a standardised dict."""
        if not usage_metadata:
            return None
        usage: Dict[str, int] = {}
        if getattr(usage_metadata, "prompt_token_count", None) is not None:
            usage["prompt_tokens"] = usage_metadata.prompt_token_count
        if getattr(usage_metadata, "candidates_token_count", None) is not None:
            usage["completion_tokens"] = usage_metadata.candidates_token_count
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
            return usage
        return None

    def _get_empty_response_retry_count(self) -> int:
        """Resolve empty-response retry count with backward-compatible fallback."""
        configured = self.valves.EMPTY_RESPONSE_RETRY_COUNT
        if configured == -1:
            self.log.warning(
                "GOOGLE_EMPTY_RESPONSE_RETRY_COUNT=-1 is deprecated. "
                "Please set a non-negative integer for independent empty-response retries. "
                "Falling back to RETRY_COUNT for compatibility."
            )
            return max(0, self.valves.RETRY_COUNT)
        if configured < -1:
            self.log.warning(
                f"Invalid GOOGLE_EMPTY_RESPONSE_RETRY_COUNT={configured}. "
                "Using 0 (disabled)."
            )
            return 0
        return configured

    def _get_safety_block_message(self, response: Any) -> Optional[str]:
        """Check for safety blocks and return appropriate message."""
        # Check prompt feedback
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[Blocked due to Prompt Safety: {response.prompt_feedback.block_reason.name}]"

        # Check candidates
        if not response.candidates:
            return "[Blocked by safety settings or no candidates generated]"

        # Check candidate finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason == types.FinishReason.SAFETY:
            blocking_rating = next(
                (r for r in candidate.safety_ratings if r.blocked), None
            )
            reason = f" ({blocking_rating.category.name})" if blocking_rating else ""
            return f"[Blocked by safety settings{reason}]"
        elif candidate.finish_reason == types.FinishReason.PROHIBITED_CONTENT:
            return "[Content blocked due to prohibited content policy violation]"

        return None

    def _is_quota_or_rate_limit_error(self, error: Exception) -> bool:
        """Return True if error indicates quota exhaustion or rate limiting."""
        status_code = getattr(error, "status_code", None)
        code = getattr(error, "code", None)
        if status_code == 429 or code == 429:
            return True

        message = str(error).lower()
        markers = (
            "quota",
            "rate limit",
            "resource_exhausted",
            "too many requests",
            "429",
            "exceeded",
        )
        return any(marker in message for marker in markers)

    def _extract_error_status_code(self, error: Exception) -> Optional[int]:
        """Best-effort extraction of HTTP/API status code from Gemini errors."""
        raw_status = getattr(error, "status_code", None)
        if isinstance(raw_status, int):
            return raw_status

        raw_code = getattr(error, "code", None)
        if isinstance(raw_code, int):
            return raw_code

        message = str(error)
        match = re.search(r"\b(429|500|502|503|504)\b", message)
        if match:
            return int(match.group(1))
        return None

    def _is_service_unavailable_error(self, error: Exception) -> bool:
        """Return True for transient 503/unavailable/high-demand errors."""
        status_code = self._extract_error_status_code(error)
        if status_code == 503:
            return True

        message = str(error).lower()
        markers = (
            "service unavailable",
            "unavailable",
            "currently experiencing high demand",
            "try again later",
            "503",
        )
        return any(marker in message for marker in markers)

    def _is_retryable_api_error(self, error: Exception) -> bool:
        """Return True for API errors that are usually transient and retryable."""
        return self._is_quota_or_rate_limit_error(error)

    def _is_expected_api_error(self, error: Exception) -> bool:
        """Return True for known API error categories (retryable or user-facing handled)."""
        return self._is_retryable_api_error(
            error
        ) or self._is_service_unavailable_error(error)

    def _service_unavailable_user_message(self) -> str:
        """User-facing message for Gemini 503 overload errors."""
        return (
            "抱歉，由于当前谷歌服务器访问量过大，模型服务暂时处于繁忙状态（错误503）。\r\n建议您稍作等待，或切换至 gemini-2.5-flash / gemini-3.1-flash-lite-preview 模型以避开高峰负载。"
        )

    def _stream_timeout_user_message(self) -> str:
        """User-facing message for streaming startup timeout."""
        timeout_sec = self.valves.STREAM_START_TIMEOUT_SEC
        return (
            f"当前模型流式响应超时（>{timeout_sec:.0f}s），本次请求已终止，避免长时间卡住。\r\n"
            "建议重试一次，或切换至 gemini-2.5-flash / gemini-3.1-flash-lite-preview。"
        )

    def _compact_error_message(self, error: Exception, max_len: int = 260) -> str:
        """Create concise one-line error message to avoid noisy logs."""
        status_code = self._extract_error_status_code(error)
        message = " ".join(str(error).split())
        if len(message) > max_len:
            message = message[: max_len - 1].rstrip() + "…"

        if status_code is not None:
            return f"{type(error).__name__}(status={status_code}): {message}"
        return f"{type(error).__name__}: {message}"

    async def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Retry a function with exponential backoff.

        Args:
            func: Async function to retry
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result from the function

        Raises:
            The last exception encountered after all retries
        """
        max_retries = max(0, self.valves.RETRY_COUNT)
        server_retry_count = 0
        key_failover_count = 0
        max_key_failovers = max(0, len(self._get_api_keys()) - 1)

        while True:
            try:
                return await func(*args, **kwargs)
            except ServerError as e:
                if self._is_service_unavailable_error(e):
                    self.log.warning(
                        "Gemini service unavailable (503) detected; skip retry. "
                        f"{self._compact_error_message(e)}"
                    )
                    raise
                # These errors might be temporary, so retry
                if server_retry_count < max_retries:
                    server_retry_count += 1
                    # Calculate backoff time (exponential with jitter)
                    wait_time = min(
                        2**server_retry_count + (0.1 * server_retry_count), 10
                    )
                    self.log.warning(
                        "Temporary server error from Google API. "
                        f"Retrying in {wait_time:.1f}s ({server_retry_count}/{max_retries}). "
                        f"{self._compact_error_message(e)}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except (ClientError, APIError) as e:
                if self._is_quota_or_rate_limit_error(e) and key_failover_count < max_key_failovers:
                    key_failover_count += 1
                    self.log.warning(
                        "Quota/rate-limit error detected. "
                        f"Switching to next API key ({key_failover_count}/{max_key_failovers}). "
                        f"{self._compact_error_message(e)}"
                    )
                    continue
                else:
                    raise
            except Exception as e:
                error_name = type(e).__name__
                if error_name in ("ClientConnectorError", "ClientOSError", "ServerDisconnectedError", "ServerTimeoutError", "TimeoutError") or isinstance(e, asyncio.TimeoutError):
                    if server_retry_count < max_retries:
                        server_retry_count += 1
                        wait_time = min(2**server_retry_count + (0.1 * server_retry_count), 10)
                        self.log.warning(
                            f"Network error ({error_name}) from Google API. "
                            f"Retrying in {wait_time:.1f}s ({server_retry_count}/{max_retries}). "
                            f"{self._compact_error_message(e)}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                # Don't retry other exceptions
                raise

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Callable,
        __tools__: dict[str, Any] | None,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        """
        Main method for sending requests to the Google Gemini endpoint.

        Args:
            body: The request body containing messages and other parameters.
            __metadata__: Request metadata
            __event_emitter__: Event emitter for status updates
            __tools__: Available tools
            __request__: FastAPI request object (for image upload)
            __user__: User information (for image upload)

        Returns:
            Response from Google Gemini API, which could be a string or an iterator for streaming.
        """
        # Setup logging for this request
        request_id = id(body)
        self.log.debug(f"Processing request {request_id}")
        self.log.debug(f"User request body: {__user__}")
        self.user = Users.get_user_by_id(__user__["id"])

        try:
            # Parse and validate model ID
            model_id = body.get("model", "")
            force_google_search_tool = False
            try:
                model_id = self._prepare_model_id(model_id)
                if model_id == "gemini-2.5-flash-search":
                    model_id = "gemini-2.5-flash"
                    force_google_search_tool = True
                self.log.debug(f"Using model: {model_id}")
            except ValueError as ve:
                return f"Model Error: {ve}"

            # Get stream flag
            stream_raw = body.get("stream", None)
            if stream_raw is None:
                stream = self.valves.STREAMING_ENABLED
            elif isinstance(stream_raw, str):
                stream = stream_raw.strip().lower() == "true"
            else:
                stream = bool(stream_raw)
            if not self.valves.STREAMING_ENABLED:
                if stream:
                    self.log.debug("Streaming disabled via GOOGLE_STREAMING_ENABLED")
                stream = False
            messages = body.get("messages", [])

            # Prepare content and extract system message
            contents, system_instruction = self._prepare_content(messages)
            if not contents:
                return "Error: No valid message content found"

            # Configure generation parameters and safety settings
            generation_config = self._configure_generation(
                body,
                system_instruction,
                __metadata__,
                __tools__,
                __user__,
                model_id,
                force_google_search_tool,
            )
            include_thoughts_enabled = self._resolve_include_thoughts(body)

            if stream:
                try:

                    stream_clients: list[genai.Client] = []

                    async def get_streaming_response():
                        client = self._get_client()
                        stream_clients.append(client)
                        return await client.aio.models.generate_content_stream(
                            model=model_id,
                            contents=contents,
                            config=generation_config,
                        )

                    async def stream_with_client_lifecycle() -> AsyncIterator[
                        Union[str, Dict[str, Any]]
                    ]:
                        try:
                            stream_start_timeout = float(
                                self.valves.STREAM_START_TIMEOUT_SEC
                            )
                            if stream_start_timeout > 0:
                                response_iterator = await asyncio.wait_for(
                                    self._retry_with_backoff(get_streaming_response),
                                    timeout=stream_start_timeout,
                                )
                            else:
                                response_iterator = await self._retry_with_backoff(
                                    get_streaming_response
                                )
                            self.log.debug(f"Request {request_id}: Got streaming response")
                            stream_response_generator = self._handle_streaming_response(
                                response_iterator,
                                __event_emitter__,
                                __request__,
                                __user__,
                                retry_stream_factory=get_streaming_response,
                                max_empty_retries=self._get_empty_response_retry_count(),
                                max_stream_error_retries=max(0, self.valves.RETRY_COUNT),
                            )
                            stream_response_iterator = stream_response_generator.__aiter__()

                            if stream_start_timeout > 0 and not include_thoughts_enabled:
                                try:
                                    first_item = await asyncio.wait_for(
                                        stream_response_iterator.__anext__(),
                                        timeout=stream_start_timeout,
                                    )
                                    yield first_item
                                except StopAsyncIteration:
                                    return
                                except asyncio.TimeoutError:
                                    self.log.warning(
                                        f"Streaming request {request_id} produced no output within timeout. "
                                        f"timeout={stream_start_timeout}s"
                                    )
                                    try:
                                        await stream_response_generator.aclose()
                                    except Exception as close_error:
                                        self.log.debug(
                                            f"Failed to close timed-out streaming generator: {close_error}"
                                        )

                                    message = self._stream_timeout_user_message()
                                    if __event_emitter__:
                                        try:
                                            await __event_emitter__(
                                                {
                                                    "type": "chat:finish",
                                                    "data": {
                                                        "role": "assistant",
                                                        "content": message,
                                                        "done": True,
                                                        "error": True,
                                                    },
                                                }
                                            )
                                        except Exception as emit_error:
                                            self.log.warning(
                                                f"Failed to emit timed-out streaming finish event: {emit_error}"
                                            )
                                    yield message
                                    return
                            elif stream_start_timeout > 0 and include_thoughts_enabled:
                                self.log.debug(
                                    f"Skipping first-yield timeout for request {request_id} because include_thoughts is enabled"
                                )

                            async for item in stream_response_iterator:
                                yield item
                        except Exception as e:
                            is_expected_api_error = isinstance(
                                e, (ClientError, ServerError, APIError)
                            ) and self._is_expected_api_error(e)
                            if is_expected_api_error:
                                self.log.warning(
                                    f"Streaming request {request_id} failed before first chunk: "
                                    f"{self._compact_error_message(e)}"
                                )
                            else:
                                self.log.exception(
                                    f"Error in streaming request {request_id}: {e}"
                                )

                            if self._is_service_unavailable_error(e):
                                message = self._service_unavailable_user_message()
                            elif isinstance(e, asyncio.TimeoutError):
                                self.log.warning(
                                    f"Streaming request {request_id} timed out before first chunk. "
                                    f"timeout={self.valves.STREAM_START_TIMEOUT_SEC}s"
                                )
                                message = self._stream_timeout_user_message()
                            else:
                                message = f"Error during streaming: {e}"

                            if __event_emitter__:
                                try:
                                    await __event_emitter__(
                                        {
                                            "type": "chat:start",
                                            "data": {"role": "assistant"},
                                        }
                                    )
                                    await __event_emitter__(
                                        {
                                            "type": "chat:finish",
                                            "data": {
                                                "role": "assistant",
                                                "content": message,
                                                "done": True,
                                                "error": True,
                                            },
                                        }
                                    )
                                except Exception as emit_error:
                                    self.log.warning(
                                        f"Failed to emit streaming error event: {emit_error}"
                                    )
                            yield message
                            return
                        finally:
                            stream_clients.clear()

                    return stream_with_client_lifecycle()

                except Exception as e:
                    is_expected_api_error = isinstance(
                        e, (ClientError, ServerError, APIError)
                    ) and self._is_expected_api_error(e)
                    if is_expected_api_error:
                        self.log.warning(
                            f"Streaming request {request_id} failed with transient API error: "
                            f"{self._compact_error_message(e)}"
                        )
                    else:
                        self.log.exception(
                            f"Error in streaming request {request_id}: {e}"
                        )
                    if self._is_service_unavailable_error(e):
                        return self._service_unavailable_user_message()
                    return f"Error during streaming: {e}"

            # Non-streaming path
            else:
                try:

                    async def get_response():
                        client = self._get_client()
                        return await client.aio.models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=generation_config,
                        )

                    empty_retry_count = 0
                    max_empty_retries = self._get_empty_response_retry_count()

                    while True:
                        # Measure duration for non-streaming path (no status to avoid false indicators)
                        start_ts = time.time()

                        response = await self._retry_with_backoff(get_response)

                        # Handle "Thinking" and produce final formatted content
                        # Check for safety blocks first
                        safety_message = self._get_safety_block_message(response)
                        if safety_message:
                            return safety_message

                        # Get the first candidate (safety checks passed)
                        candidate = response.candidates[0]

                        # Process content parts - use new streamlined approach
                        parts = getattr(getattr(candidate, "content", None), "parts", [])
                        if not parts:
                            if empty_retry_count < max_empty_retries:
                                empty_retry_count += 1
                                self.log.warning(
                                    f"Non-streaming returned empty parts. Retrying ({empty_retry_count}/{max_empty_retries})"
                                )
                                continue
                            return "[No content generated or unexpected response structure]"

                        answer_segments: list[str] = []
                        thought_segments: list[str] = []

                        for part in parts:
                            if getattr(part, "thought", False) and getattr(
                                part, "text", None
                            ):
                                thought_segments.append(part.text)
                            elif getattr(part, "text", None):
                                answer_segments.append(part.text)

                        final_answer = "".join(answer_segments)

                        # Apply grounding (if available) and send sources/status as needed
                        grounding_metadata_list = []
                        if getattr(candidate, "grounding_metadata", None):
                            grounding_metadata_list.append(candidate.grounding_metadata)
                        if grounding_metadata_list:
                            cited = await self._process_grounding_metadata(
                                grounding_metadata_list,
                                final_answer,
                                __event_emitter__,
                            )
                            final_answer = cited or final_answer

                        # Combine all content
                        full_response = ""

                        # If we have thoughts, wrap them using <details>
                        if thought_segments:
                            duration_s = int(max(0, time.time() - start_ts))
                            # Format each line with > for blockquote
                            thought_content = "".join(thought_segments).strip()
                            quoted_lines = []
                            for line in thought_content.split("\n"):
                                quoted_lines.append(f"> {line}")
                            quoted_content = "\n".join(quoted_lines)

                            details_block = f"""<details>
<summary>思考过程 ({duration_s}s)</summary>

{quoted_content}

</details>"""
                            full_response += details_block
                            full_response += "\n\n"

                        # Add the main answer
                        full_response += final_answer

                        if not final_answer.strip() and empty_retry_count < max_empty_retries:
                            empty_retry_count += 1
                            self.log.warning(
                                f"Non-streaming returned empty content. Retrying ({empty_retry_count}/{max_empty_retries})"
                            )
                            continue
                        elif not final_answer.strip() and max_empty_retries > 0:
                            warning_msg = "⚠️ [模型多次尝试后未生成有效正文内容，可能由于安全策略拦截或当前无法作答]"
                            full_response = (full_response + f"\n\n{warning_msg}").strip()

                        # Build response with usage for middleware to extract and save to DB
                        usage = self._build_usage_dict(
                            getattr(response, "usage_metadata", None)
                        )

                        content = (
                            full_response if full_response else "[No content generated]"
                        )
                        result = {
                            "choices": [
                                {"message": {"role": "assistant", "content": content}}
                            ],
                        }
                        if usage:
                            result["usage"] = usage
                        return result

                except Exception as e:
                    is_expected_api_error = isinstance(
                        e, (ClientError, ServerError, APIError)
                    ) and self._is_expected_api_error(e)
                    if is_expected_api_error:
                        self.log.warning(
                            f"Non-streaming request {request_id} failed with transient API error: "
                            f"{self._compact_error_message(e)}"
                        )
                    else:
                        self.log.exception(
                            f"Error in non-streaming request {request_id}: {e}"
                        )
                    if self._is_service_unavailable_error(e):
                        return self._service_unavailable_user_message()
                    return f"Error generating content: {e}"

        except (ClientError, ServerError, APIError) as api_error:
            if self._is_service_unavailable_error(api_error):
                return self._service_unavailable_user_message()
            error_type = type(api_error).__name__
            error_msg = f"{error_type}: {api_error}"
            self.log.error(error_msg)
            return error_msg

        except ValueError as ve:
            error_msg = f"Configuration error: {ve}"
            self.log.error(error_msg)
            return error_msg

        except Exception as e:
            # Log the full error with traceback
            import traceback

            error_trace = traceback.format_exc()
            self.log.exception(f"Unexpected error: {e}\n{error_trace}")

            # Return a user-friendly error message
            return f"An error occurred while processing your request: {e}"
