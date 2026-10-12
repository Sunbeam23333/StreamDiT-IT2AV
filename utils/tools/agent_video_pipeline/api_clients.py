from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from openai import OpenAI


DEFAULT_API_BASE = os.environ.get("VENUS_API_BASE", "")
logger = logging.getLogger(__name__)


def _get_mime_type(file_path: str | Path) -> str:
    guessed, _ = mimetypes.guess_type(str(file_path))
    return guessed or "application/octet-stream"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
                elif "text" in item:
                    texts.append(str(item.get("text", "")))
        return "\n".join(text for text in texts if text)
    return str(content or "")


def _decode_data_url(raw: str) -> bytes:
    if "," not in raw:
        return base64.b64decode(raw)
    return base64.b64decode(raw.split(",", 1)[1])


def _summarize_payload(value: Any, max_len: int = 240) -> str:
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in list(value.items())[:8]:
            if isinstance(item, str):
                summary[key] = item[:max_len] + ("..." if len(item) > max_len else "")
            elif isinstance(item, (dict, list)):
                summary[key] = f"<{type(item).__name__}>"
            else:
                summary[key] = item
        return json.dumps(summary, ensure_ascii=False)
    if isinstance(value, list):
        parts = [_summarize_payload(item, max_len=max_len) for item in value[:3]]
        if len(value) > 3:
            parts.append("...")
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, str):
        return value[:max_len] + ("..." if len(value) > max_len else "")
    return str(value)


def _collect_image_data_urls(value: Any, results: list[str] | None = None) -> list[str]:
    if results is None:
        results = []
    if isinstance(value, dict):
        maybe_url = value.get("url")
        if isinstance(maybe_url, str) and maybe_url.startswith("data:image"):
            results.append(maybe_url)
        for key in ("b64_json", "image_base64", "imageBase64", "base64", "data"):
            maybe_b64 = value.get(key)
            if isinstance(maybe_b64, str) and maybe_b64:
                mime_type = value.get("mimeType") if isinstance(value.get("mimeType"), str) else "image/png"
                if not maybe_b64.startswith("data:"):
                    results.append(f"data:{mime_type};base64,{maybe_b64}")
        for item in value.values():
            _collect_image_data_urls(item, results)
    elif isinstance(value, list):
        for item in value:
            _collect_image_data_urls(item, results)
    elif isinstance(value, str) and value.startswith("data:image"):
        results.append(value)
    return results


class VenusProxyClient:
    """Thin wrapper for Venus llmproxy/chatproxy multimodal APIs."""

    def __init__(
        self,
        proxy_token: str,
        api_base: str = DEFAULT_API_BASE,
        timeout: int = 180,
    ):
        self.proxy_token = proxy_token
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.openai_client = OpenAI(base_url=f"{self.api_base}/chatproxy", api_key=proxy_token)

    @classmethod
    def from_env(
        cls,
        api_base: str = DEFAULT_API_BASE,
        image_app_group_id: int = int(os.environ.get("VENUS_APP_GROUP_ID", "0") or 0),
        timeout: int = 180,
    ) -> "VenusProxyClient | None":
        explicit = os.getenv("VENUS_PROXY_TOKEN") or os.getenv("VENUS_LLM_PROXY_TOKEN")
        secret_id = os.getenv("ENV_VENUS_OPENAPI_SECRET_ID")
        proxy_token = explicit
        if not proxy_token and secret_id:
            proxy_token = secret_id if "@" in secret_id else f"{secret_id}@{image_app_group_id}"
        if not proxy_token:
            return None
        return cls(proxy_token=proxy_token, api_base=api_base, timeout=timeout)

    @property
    def llm_url(self) -> str:
        return f"{self.api_base}/llmproxy/chat/completions"

    @property
    def storage_info_url(self) -> str:
        return f"{self.api_base}/chat/temp/storage/info"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.proxy_token}",
        }

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "gemini-3-pro",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        fallback_models: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        candidates: list[str] = [model]
        for fallback in fallback_models or []:
            if fallback and fallback not in candidates:
                candidates.append(fallback)

        last_error: str | None = None
        for candidate in candidates:
            payload = {
                "model": candidate,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            response = self.session.post(
                self.llm_url,
                headers=self._headers(),
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                data["_resolved_model"] = candidate
                return data
            last_error = response.text
        raise RuntimeError(f"Gemini/Venus LLM 调用失败: {last_error}")

    def chat_text(
        self,
        messages: list[dict[str, Any]],
        model: str = "gemini-3-pro",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        fallback_models: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        data = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback_models=fallback_models,
            **kwargs,
        )
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _extract_text_content(content), str(data.get("_resolved_model", model))

    def generate_image(
        self,
        prompt: str,
        output_path: str | Path,
        model: str = "gemini-3-pro-image",
        size: str = "1024x1024",
        reference_image_paths: Iterable[str | Path] | None = None,
        fallback_models: Iterable[str] | None = None,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        reference_paths = [Path(path) for path in (reference_image_paths or []) if Path(path).exists()]

        candidate_models: list[str] = [model]
        for fallback in fallback_models or []:
            if fallback and fallback not in candidate_models:
                candidate_models.append(fallback)
        if "gpt-image-1" not in candidate_models:
            candidate_models.append("gpt-image-1")

        last_error: Exception | None = None
        for candidate in candidate_models:
            attempts = 1 if candidate.startswith("gpt-image") else 2
            for attempt in range(1, attempts + 1):
                try:
                    logger.info("[VenusProxy] image generate model=%s attempt=%d", candidate, attempt)
                    if candidate.startswith("gpt-image"):
                        self._generate_gpt_image(prompt=prompt, output_path=output, model=candidate, size=size)
                    else:
                        self._generate_gemini_image(
                            prompt=prompt,
                            output_path=output,
                            model=candidate,
                            reference_image_paths=reference_paths,
                        )
                    logger.info("[VenusProxy] image generate done model=%s", candidate)
                    return output
                except Exception as exc:
                    last_error = exc
                    logger.warning("[VenusProxy] image generate failed model=%s attempt=%d error=%s", candidate, attempt, exc)
        raise RuntimeError(f"Image generation failed after fallbacks: {last_error}")

    def _generate_gpt_image(self, prompt: str, output_path: Path, model: str, size: str) -> None:
        response = self.openai_client.images.generate(
            prompt=prompt,
            model=model,
            n=1,
            size=size,
        )
        raw = response.data[0].b64_json
        output_path.write_bytes(base64.b64decode(raw))

    def _generate_gemini_image(
        self,
        prompt: str,
        output_path: Path,
        model: str,
        reference_image_paths: Iterable[Path],
    ) -> None:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in reference_image_paths:
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            mime_type = _get_mime_type(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000,
        }
        response = self.session.post(
            self.llm_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini image 调用失败: {response.text}")
        data = response.json()
        image_candidates = _collect_image_data_urls(data)
        for image_url in image_candidates:
            try:
                output_path.write_bytes(_decode_data_url(image_url))
                return
            except Exception:
                continue

        content_payload = data.get("choices", [{}])[0].get("message", {}).get("content", [])
        raise RuntimeError(
            "Gemini image 返回中未找到可保存的图片数据; "
            f"content_summary={_summarize_payload(content_payload)}"
        )

    def get_temp_storage_info(self, model: str, filename: str) -> dict[str, Any]:
        payload = {"model": model, "filename": filename}
        response = self.session.post(
            self.storage_info_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"获取临时存储信息失败: {response.text}")
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"获取临时存储信息失败: {result}")
        return result["data"]

    def upload_file_to_storage(self, file_path: str | Path, put_url: str, mime_type: str | None = None) -> None:
        import urllib3

        file_path = Path(file_path)
        mime_type = mime_type or _get_mime_type(file_path)
        file_bytes = file_path.read_bytes()
        is_gcs = "storage.googleapis.com" in put_url or "storage.cloud.google.com" in put_url

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        headers: dict[str, str] = {} if is_gcs else {"Content-Type": mime_type}
        response = http.request("PUT", put_url, body=file_bytes, headers=headers, timeout=urllib3.Timeout(total=300))
        if response.status not in {200, 201}:
            raise RuntimeError(f"上传文件失败: HTTP {response.status}, body={response.data[:300]}")

    def analyze_file(
        self,
        file_path: str | Path,
        prompt: str,
        model: str = "gemini-2.5-flash",
    ) -> str:
        return self.analyze_files(file_paths=[file_path], prompt=prompt, model=model)

    def analyze_files(
        self,
        file_paths: Iterable[str | Path],
        prompt: str,
        model: str = "gemini-2.5-flash",
    ) -> str:
        normalized_paths = [Path(path) for path in file_paths]
        if normalized_paths and all(_get_mime_type(path).startswith("image/") for path in normalized_paths):
            logger.info("[VenusProxy] analyze inline-images model=%s files=%d", model, len(normalized_paths))
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for file_path in normalized_paths:
                mime_type = _get_mime_type(file_path)
                encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    }
                )
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4000,
            }
            response = self.session.post(
                self.llm_url,
                headers=self._headers(),
                data=json.dumps(payload),
                timeout=300,
            )
            logger.info("[VenusProxy] analyze response status=%s files=%d", response.status_code, len(normalized_paths))
            if response.status_code != 200:
                raise RuntimeError(f"Gemini 多模态核验失败: {response.text}")
            result = response.json()
            return _extract_text_content(result.get("choices", [{}])[0].get("message", {}).get("content", ""))

        uploaded_items: list[dict[str, Any]] = []
        for file_path in normalized_paths:
            mime_type = _get_mime_type(file_path)
            logger.info("[VenusProxy] analyze start model=%s file=%s", model, file_path.name)
            storage_info = self.get_temp_storage_info(model=model, filename=file_path.name)
            logger.info("[VenusProxy] analyze storage ready bucket=%s path=%s", storage_info["bucketName"], storage_info["filepath"])
            self.upload_file_to_storage(file_path, storage_info["putUrl"], mime_type=mime_type)
            logger.info("[VenusProxy] analyze upload done file=%s", file_path.name)
            uploaded_items.append(
                {
                    "type": "gemini_multimodal_url",
                    "gemini_multimodal_url": {
                        "bucketName": storage_info["bucketName"],
                        "filepath": storage_info["filepath"],
                        "mimeType": mime_type,
                    },
                }
            )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *uploaded_items],
                }
            ],
            "max_tokens": 4000,
        }
        response = self.session.post(
            self.llm_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=300,
        )
        logger.info("[VenusProxy] analyze response status=%s files=%d", response.status_code, len(uploaded_items))
        if response.status_code != 200:
            raise RuntimeError(f"Gemini 多模态核验失败: {response.text}")
        result = response.json()
        return _extract_text_content(result.get("choices", [{}])[0].get("message", {}).get("content", ""))


class VenusVideoClient:
    """Signed Venus OpenAPI client for Veo image-to-video generation."""

    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        app_group_id: int = int(os.environ.get("VENUS_VIDEO_APP_GROUP_ID", "0") or 0),
        api_base: str = DEFAULT_API_BASE,
        read_timeout: int = 10,
    ):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.app_group_id = app_group_id
        self.api_base = api_base.rstrip("/")
        self.read_timeout = read_timeout

    @classmethod
    def from_env(
        cls,
        app_group_id: int = int(os.environ.get("VENUS_VIDEO_APP_GROUP_ID", "0") or 0),
        api_base: str = DEFAULT_API_BASE,
    ) -> "VenusVideoClient | None":
        secret_id = os.getenv("ENV_VENUS_OPENAPI_SECRET_ID")
        secret_key = os.getenv("ENV_VENUS_OPENAPI_SECRET_KEY") or secret_id
        if not secret_id and secret_key:
            secret_id = secret_key
        app_group_env = os.getenv("VENUS_VIDEO_APP_GROUP_ID")
        final_app_group = int(app_group_env) if app_group_env else app_group_id
        if not secret_id or not secret_key:
            return None
        return cls(secret_id=secret_id, secret_key=secret_key, app_group_id=final_app_group, api_base=api_base)

    @property
    def available(self) -> bool:
        try:
            import venus_api_base  # noqa: F401
        except Exception:
            return False
        return bool(self.secret_id and self.secret_key and self.app_group_id)

    def _client(self):
        from venus_api_base.config import Config
        from venus_api_base.http_client import HttpClient

        return HttpClient(
            config=Config(read_timeout=self.read_timeout),
            secret_id=self.secret_id,
            secret_key=self.secret_key,
        )

    def _submit(self, payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
        client = self._client()
        response = client.post(
            url=f"{self.api_base}{endpoint}",
            header={"Content-Type": "application/json"},
            body=json.dumps(payload),
        )
        if response.get("code") != 0:
            raise RuntimeError(f"视频任务提交失败: {response}")
        return response

    def _query(self, task_id: int) -> dict[str, Any]:
        client = self._client()
        response = client.get(url=f"{self.api_base}/video/query?task_id={task_id}", header={})
        if response.get("code") != 0:
            raise RuntimeError(f"视频任务查询失败: {response}")
        return response

    def _poll(self, task_id: int, timeout_seconds: int = 1800, poll_interval: int = 20) -> dict[str, Any]:
        start = time.time()
        while True:
            result = self._query(task_id)
            status = result.get("data", {}).get("status")
            if status in {"succeed", "failed"}:
                return result
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"视频任务轮询超时: {task_id}")
            time.sleep(poll_interval)

    def generate_veo_image2video(
        self,
        prompt: str,
        first_frame_path: str | Path,
        output_path: str | Path,
        last_frame_path: str | Path | None = None,
        model: str = "veo-3.1-fast",
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
        generate_audio: bool = False,
        duration_s: float = 3.0,
    ) -> Path:
        if not self.available:
            raise RuntimeError("Veo 视频客户端不可用：缺少 SDK 或签名凭据")

        first_frame = Path(first_frame_path)
        requested_duration = max(2, int(round(duration_s)))
        if requested_duration != 8:
            logger.info("[VenusVideo] Veo 当前仅支持 8 秒输出，请求 %ss，将在生成后由本地链路裁切", requested_duration)
        payload: dict[str, Any] = {
            "model": model,
            "app_group_id": self.app_group_id,
            "prompt": prompt,
            "image": {
                "mimeType": _get_mime_type(first_frame),
                "bytesBase64Encoded": base64.b64encode(first_frame.read_bytes()).decode("utf-8"),
            },
            "durationSeconds": 8,
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "enhancePrompt": True,
            "personGeneration": "allow_adult",
            "generateAudio": generate_audio,
        }
        if last_frame_path and Path(last_frame_path).exists():
            last_frame = Path(last_frame_path)
            payload["lastFrame"] = {
                "mimeType": _get_mime_type(last_frame),
                "bytesBase64Encoded": base64.b64encode(last_frame.read_bytes()).decode("utf-8"),
            }

        submit_result = self._submit(payload, endpoint="/video/image2video")
        task_id = int(submit_result["data"]["task_id"])
        final_result = self._poll(task_id)
        status = final_result.get("data", {}).get("status")
        if status != "succeed":
            raise RuntimeError(f"Veo 视频生成失败: {final_result}")
        urls = final_result.get("data", {}).get("response_attachment_urls", [])
        if not urls:
            raise RuntimeError(f"Veo 视频生成成功但没有返回下载链接: {final_result}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(urls[0], timeout=300)
        response.raise_for_status()
        output.write_bytes(response.content)
        return output
