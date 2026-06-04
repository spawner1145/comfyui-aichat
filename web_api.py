import logging
import httpx

from .channel_config import load_channel_groups, resolve_channel_config, save_channel_groups

logger = logging.getLogger("aichat_web_api")

try:
    from aiohttp import web
    from server import PromptServer

    _HAS_SERVER = PromptServer.instance is not None
except Exception as e:
    logger.warning(f"[aichat] 未能导入 PromptServer / aiohttp，接口已禁用: {e}")
    _HAS_SERVER = False

BASE = "/aichat/api/"


def _build_proxies(proxy_http: str, proxy_https: str):
    proxies = {}
    if proxy_http and proxy_http.strip():
        proxies["http://"] = proxy_http.strip()
    if proxy_https and proxy_https.strip():
        proxies["https://"] = proxy_https.strip()
    return proxies or None


def _make_async_client(timeout: float, proxies, **kwargs) -> "httpx.AsyncClient":
    if proxies:
        return httpx.AsyncClient(proxies=proxies, timeout=timeout, **kwargs)
    return httpx.AsyncClient(timeout=timeout, **kwargs)


async def _read_json(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


async def _read_common_params(request):
    data = await _read_json(request)
    try:
        timeout = float(data.get("timeout") or 60.0)
    except (TypeError, ValueError):
        timeout = 60.0
    return {
        "provider": (data.get("provider") or "").strip().lower(),
        "channel_group": (data.get("channel_group") or "").strip(),
        "api_key": (data.get("api_key") or "").strip(),
        "base_url": (data.get("base_url") or "").strip(),
        "proxy_http": data.get("proxy_http") or "",
        "proxy_https": data.get("proxy_https") or "",
        "timeout": timeout,
    }


def _extract_openai_models(body) -> list:
    items = []
    if isinstance(body, dict):
        items = body.get("data") or body.get("models") or []
    elif isinstance(body, list):
        items = body

    out = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            mid = it.get("id") or it.get("name") or it.get("model")
            if mid:
                out.append(str(mid))
    return sorted(set(out))


if _HAS_SERVER:

    @PromptServer.instance.routes.post(BASE + "groups/get")
    async def _aichat_groups_get(request):
        data = await _read_json(request)
        provider = (data.get("provider") or "").strip().lower()
        try:
            groups = load_channel_groups(provider)
            return web.json_response({"groups": groups, "error": None})
        except Exception as e:
            logger.error(f"[aichat] groups/get 失败: {type(e).__name__} - {e}")
            return web.json_response({"groups": [], "error": f"{type(e).__name__}: {e}"})

    @PromptServer.instance.routes.post(BASE + "groups/save")
    async def _aichat_groups_save(request):
        data = await _read_json(request)
        provider = (data.get("provider") or "").strip().lower()
        groups = data.get("groups") or []
        try:
            saved = save_channel_groups(provider, groups)
            return web.json_response({"groups": saved, "error": None})
        except Exception as e:
            logger.error(f"[aichat] groups/save 失败: {type(e).__name__} - {e}")
            return web.json_response({"groups": [], "error": f"{type(e).__name__}: {e}"})

    @PromptServer.instance.routes.post(BASE + "openai/get_models")
    async def _aichat_openai_get_models(request):
        p = await _read_common_params(request)
        try:
            api_key, base_url, _ = resolve_channel_config(
                "openai",
                channel_group=p["channel_group"],
                api_key=p["api_key"],
                base_url=p["base_url"],
            )
        except Exception as e:
            return web.json_response({"models": [], "error": str(e)})

        if not api_key:
            return web.json_response({"models": [], "error": "API Key 不能为空"})
        if not base_url:
            return web.json_response({"models": [], "error": "Base URL 不能为空"})

        if not base_url.endswith("/"):
            base_url += "/"
        proxies = _build_proxies(p["proxy_http"], p["proxy_https"])
        url = base_url + "models"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with _make_async_client(p["timeout"], proxies) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                snippet = resp.text[:300] if resp.text else ""
                return web.json_response({"models": [], "error": f"HTTP {resp.status_code}: {snippet}"})
            models = _extract_openai_models(resp.json())
            logger.info(f"[aichat] OpenAI 获取到 {len(models)} 个模型 ({base_url})")
            return web.json_response({"models": models, "error": None})
        except Exception as e:
            logger.error(f"[aichat] OpenAI get_models 失败: {type(e).__name__} - {e}")
            return web.json_response({"models": [], "error": f"{type(e).__name__}: {e}"})

    @PromptServer.instance.routes.post(BASE + "gemini/get_models")
    async def _aichat_gemini_get_models(request):
        p = await _read_common_params(request)
        try:
            api_key, base_url, _ = resolve_channel_config(
                "gemini",
                channel_group=p["channel_group"],
                api_key=p["api_key"],
                base_url=p["base_url"],
            )
        except Exception as e:
            return web.json_response({"models": [], "error": str(e)})

        if not api_key:
            return web.json_response({"models": [], "error": "API Key 不能为空"})
        if not base_url:
            return web.json_response({"models": [], "error": "Base URL 不能为空"})

        base_url = base_url.rstrip("/")
        proxies = _build_proxies(p["proxy_http"], p["proxy_https"])
        try:
            async with _make_async_client(
                p["timeout"], proxies, base_url=base_url, params={"key": api_key}
            ) as client:
                models = []
                page_token = None
                for _ in range(20):
                    params = {"pageSize": 1000}
                    if page_token:
                        params["pageToken"] = page_token
                    resp = await client.get("/v1beta/models", params=params)
                    if resp.status_code != 200:
                        snippet = resp.text[:300] if resp.text else ""
                        return web.json_response({"models": [], "error": f"HTTP {resp.status_code}: {snippet}"})
                    body = resp.json()
                    for m in body.get("models", []):
                        name = m.get("name", "")
                        methods = m.get("supportedGenerationMethods") or []
                        if methods and not any(x in methods for x in ("generateContent", "streamGenerateContent")):
                            continue
                        if name.startswith("models/"):
                            name = name[len("models/"):]
                        if name:
                            models.append(name)
                    page_token = body.get("nextPageToken")
                    if not page_token:
                        break
                models = sorted(set(models))
                logger.info(f"[aichat] Gemini 获取到 {len(models)} 个模型 ({base_url})")
                return web.json_response({"models": models, "error": None})
        except Exception as e:
            logger.error(f"[aichat] Gemini get_models 失败: {type(e).__name__} - {e}")
            return web.json_response({"models": [], "error": f"{type(e).__name__}: {e}"})

    print(
        f"[aichat] 接口已注册: POST {BASE}groups/get | POST {BASE}groups/save | POST {BASE}openai/get_models | POST {BASE}gemini/get_models"
    )
