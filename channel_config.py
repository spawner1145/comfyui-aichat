import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROVIDER_FILES = {
    "openai": os.path.join(BASE_DIR, "openai.yaml"),
    "gemini": os.path.join(BASE_DIR, "gemini.yaml"),
}


def _ensure_provider(provider: str) -> str:
    key = (provider or "").strip().lower()
    if key not in PROVIDER_FILES:
        raise ValueError(f"不支持的 provider: {provider}")
    return key


def get_provider_file(provider: str) -> str:
    return PROVIDER_FILES[_ensure_provider(provider)]


def _normalize_group(name: str, item: Any) -> Optional[Dict[str, str]]:
    if not name or not isinstance(item, dict):
        return None
    base_url = str(item.get("base_url") or item.get("baseurl") or "").strip()
    api_key = str(item.get("api_key") or item.get("apikey") or item.get("key") or "").strip()
    if not base_url and not api_key:
        return None
    return {
        "name": name.strip(),
        "base_url": base_url,
        "api_key": api_key,
    }


def _parse_channels_data(data: Any) -> List[Dict[str, str]]:
    groups: List[Dict[str, str]] = []
    if isinstance(data, dict):
        source = data.get("channels", data)
        if isinstance(source, dict):
            for name, item in source.items():
                group = _normalize_group(str(name), item)
                if group:
                    groups.append(group)
        elif isinstance(source, list):
            for item in source:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                group = _normalize_group(name, item)
                if group:
                    groups.append(group)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            group = _normalize_group(name, item)
            if group:
                groups.append(group)
    return groups


def _dump_channels_data(groups: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    channels: Dict[str, Dict[str, str]] = {}
    for group in groups:
        name = str(group.get("name") or "").strip()
        if not name:
            continue
        channels[name] = {
            "base_url": str(group.get("base_url") or "").strip(),
            "api_key": str(group.get("api_key") or "").strip(),
        }
    return {"channels": channels}


def load_channel_groups(provider: str) -> List[Dict[str, str]]:
    path = get_provider_file(provider)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取 yaml 配置")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    groups = _parse_channels_data(data)
    groups.sort(key=lambda x: x["name"].lower())
    return groups


def save_channel_groups(provider: str, groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    path = get_provider_file(provider)
    normalized = _parse_channels_data(groups)
    normalized.sort(key=lambda x: x["name"].lower())
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法写入 yaml 配置")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _dump_channels_data(normalized),
            f,
            allow_unicode=True,
            sort_keys=False,
        )
    return normalized


def resolve_channel_config(
    provider: str,
    channel_group: str = "",
    api_key: str = "",
    base_url: str = "",
) -> Tuple[str, str, Dict[str, str]]:
    selected_group = (channel_group or "").strip()
    direct_api_key = (api_key or "").strip()
    direct_base_url = (base_url or "").strip()

    if selected_group:
        groups = load_channel_groups(provider)
        for group in groups:
            if group["name"] == selected_group:
                resolved_base_url = group["base_url"].strip()
                resolved_api_key = group["api_key"].strip()
                if not resolved_base_url:
                    raise ValueError(f"渠道组 {selected_group} 缺少 base_url")
                if not resolved_api_key:
                    raise ValueError(f"渠道组 {selected_group} 缺少 api_key")
                return resolved_api_key, resolved_base_url, group
        raise ValueError(f"未找到渠道组: {selected_group}")

    return direct_api_key, direct_base_url, {}
