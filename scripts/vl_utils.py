#!/usr/bin/env python3
"""
VL API 工具：支持多实例负载均衡（api_urls 轮询），支持每 URL 指定不同 model。
"""

from __future__ import annotations

import threading
from typing import Any


_lock = threading.Lock()
_counter = 0
# 8b 单批最多页数，超过则分两批处理
PAGE_THRESHOLD_8B_BATCH = 20


def _normalize_endpoint(item: Any, default_model: str, default_max_tokens: int) -> tuple[str, str, int]:
    """将 api_urls 中的项转为 (url, model, max_tokens)。"""
    if isinstance(item, str):
        return str(item), default_model, default_max_tokens
    if isinstance(item, dict):
        url = str(item.get("url", ""))
        model = str(item.get("model", default_model))
        max_tok = item.get("max_tokens")
        max_tokens = int(max_tok) if max_tok is not None else default_max_tokens
        return url, model, max_tokens
    return "", default_model, default_max_tokens


def get_vl_endpoint(vl: dict[str, Any], num_pages: int | None = None) -> tuple[str, str, int]:
    """
    获取本次请求应使用的 VL API (url, model, max_tokens)。
    - 若配置了 api_urls（列表），则轮询返回下一个；每项可为字符串或 {url, model?, max_tokens?}
    - 否则使用 api_url、vl.model、vl.max_tokens
    - num_pages: 保留参数供 caller 判断是否需分批（8b 时 >20 页分两批）
    """
    default_model = str(vl.get("model", ""))
    default_max_tokens = int(vl.get("max_tokens", 4096))
    urls = vl.get("api_urls")
    if urls and isinstance(urls, list) and len(urls) >= 1:
        global _counter
        with _lock:
            # 不再排除 8b：>25 页时由 caller 分批处理
            item = urls[_counter % len(urls)]
            _counter += 1
            return _normalize_endpoint(item, default_model, default_max_tokens)
    url = str(vl.get("api_url", ""))
    return url, default_model, default_max_tokens


def get_vl_url(vl: dict[str, Any]) -> str:
    """兼容旧用法：仅返回 URL（使用默认 model）。"""
    url, _, _ = get_vl_endpoint(vl)
    return url


def has_vl_config(vl: dict[str, Any]) -> bool:
    """检查 vl 配置是否有效（有 api_url 或 api_urls）。"""
    if vl.get("api_url"):
        return True
    urls = vl.get("api_urls")
    return bool(urls and isinstance(urls, list) and len(urls) >= 1)
