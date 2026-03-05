#!/usr/bin/env python3
"""
VL API 工具：支持多实例负载均衡（api_urls 轮询）。
"""

from __future__ import annotations

import threading
from typing import Any


_lock = threading.Lock()
_counter = 0


def get_vl_url(vl: dict[str, Any]) -> str:
    """
    获取本次请求应使用的 VL API URL。
    - 若配置了 api_urls（列表），则轮询返回下一个
    - 否则使用 api_url
    """
    urls = vl.get("api_urls")
    if urls and isinstance(urls, list) and len(urls) >= 1:
        global _counter
        with _lock:
            url = urls[_counter % len(urls)]
            _counter += 1
            return str(url)
    return str(vl.get("api_url", ""))


def has_vl_config(vl: dict[str, Any]) -> bool:
    """检查 vl 配置是否有效（有 api_url 或 api_urls）。"""
    if vl.get("api_url"):
        return True
    urls = vl.get("api_urls")
    return bool(urls and isinstance(urls, list) and len(urls) >= 1)
