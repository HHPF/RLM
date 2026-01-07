from typing import Any


def filter_sensitive_keys(kwargs: dict[str, Any]) -> dict[str, Any]:
    """从关键字参数中过滤掉敏感键，如 API 密钥。"""
    filtered = {}
    for key, value in kwargs.items():
        key_lower = key.lower()
        if "api" in key_lower and "key" in key_lower:
            continue
        filtered[key] = value
    return filtered
