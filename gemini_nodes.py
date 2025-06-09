import httpx
import json
import mimetypes
import asyncio
import base64
import os
from typing import AsyncGenerator, Dict, List, Optional, Union, Any, Callable
import aiofiles
import logging
import io

try:
    import torch
    import numpy as np
    from PIL import Image
    import folder_paths
    print("Loaded ComfyUI modules successfully for Gemini.")
except ImportError:
     print("Warning: Could not import ComfyUI modules (torch, numpy, PIL, folder_paths). Gemini Nodes will not work outside ComfyUI.")
     torch = None
     Image = None
     folder_paths = None

logger = logging.getLogger('Gemini_ComfyUI')
logger.setLevel(logging.INFO)

class GeminiAPI:
     def __init__(
        self,
        apikey: str,
        baseurl: str = "https://generativelanguage.googleapis.com",
        model: str = "gemini-2.0-flash-001",
        proxies: Optional[Dict[str, str]] = None,
        timeout: float = 180.0,
    ):
        if not apikey:
             raise ValueError("Gemini API Key 不能为空")
        self.apikey = apikey
        self.baseurl = baseurl.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=baseurl,
            params={'key': apikey},
            proxies=proxies,
            timeout=timeout
        )
        logger.info(f"GeminiAPI Client Initialized: model={self.model}, base_url={self.baseurl}")

     async def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, Union[str, None]]:
        """上传单个文件到 Gemini File API，并检查 ACTIVE 状态。返回可用于 fileData 的字典"""
        if not os.path.exists(file_path):
             logger.error(f"文件 {file_path} 不存在")
             raise FileNotFoundError(f"文件上传失败: 路径 {file_path} 不存在")
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 1.9 * 1024 * 1024 * 1024:  # ~1.9GB limit
                raise ValueError(f"文件 {file_path} 大小接近或超过 2GB 限制")
        except OSError as e:
             logger.error(f"获取文件 {file_path} 大小失败: {e}")
             raise

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
            logger.warning(f"无法检测文件 {file_path} 的 MIME 类型，使用默认值: {mime_type}")

        file_uri = None
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                headers = {"X-Goog-Upload-Protocol": "multipart"}
                metadata = {'file': {'displayName': display_name or os.path.basename(file_path), 'mimeType': mime_type}}
                files = {
                     'metadata': (None, json.dumps(metadata), 'application/json'),
                     'file': (os.path.basename(file_path), await f.read(), mime_type)
                 }
                logger.info(f"开始上传文件: {file_path} ({mime_type})")
                upload_url = f"{self.baseurl}/upload/v1beta/files"
                response = await self.client.post(upload_url, files=files, headers=headers)
                response.raise_for_status()
                file_data = response.json()
                logger.debug(f"文件上传原始响应: {file_data}")
                file_uri = file_data.get('file', {}).get('uri')
                if not file_uri:
                     raise ValueError(f"上传成功但未返回 file URI: {file_data}")
                logger.info(f"文件初步上传成功: {file_uri}")
                
        except Exception as e:
            logger.error(f"文件 {file_path} 上传请求失败: {type(e).__name__} - {str(e)}")
            raise RuntimeError(f"文件 {file_path} 上传请求失败: {type(e).__name__} - {str(e)}") from e

        if not await self.wait_for_file_active(file_uri, timeout=120, interval=3):
            error_msg = f"文件 {file_path} ({file_uri}) 未能在规定时间内变为 ACTIVE 状态或处理失败"
            logger.error(error_msg)
            raise TimeoutError(error_msg)

        logger.info(f"文件 {file_path} 上传并激活成功，URI: {file_uri}")
        return {"fileData": {"mimeType": mime_type, "fileUri": file_uri} }

     async def wait_for_file_active(self, file_uri: str, timeout: int = 120, interval: int = 3) -> bool:
        """等待文件状态变为 ACTIVE"""
        # URI 格式: https://generativelanguage.googleapis.com/v1beta/files/xxxxxxxx
        try:
             file_name_part = file_uri.split('/v1beta/')[-1] # files/xxxxxxxx
        except:
             logger.error(f"无法从 URI 解析文件名: {file_uri}")
             return False
        
        start_time = asyncio.get_event_loop().time()
        logger.info(f"开始检查文件状态: {file_name_part}, 超时: {timeout}s, 间隔: {interval}s")

        while (asyncio.get_event_loop().time() - start_time < timeout):
            try:
                # GET 请求不需要 /upload
                response = await self.client.get(f"/v1beta/{file_name_part}")
                response.raise_for_status()
                file_info = response.json()
                state = file_info.get('state')
                logger.info(f"文件 {file_name_part} 当前状态: {state}")
                if state == "ACTIVE":
                    return True
                elif state == "FAILED":
                    logger.error(f"文件 {file_name_part} 处理失败。 错误: {file_info.get('error')}")
                    return False
                elif state == "PROCESSING":
                     await asyncio.sleep(interval)
                else: # state is None or unknown
                     await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"检查文件 {file_name_part} 状态时出错: {type(e).__name__} - {e}, 重试中...")
                await asyncio.sleep(interval * 2) # Error, wait longer
        logger.warning(f"等待文件 {file_name_part} 状态变为 ACTIVE 超时 ({timeout}秒)")
        return False


     async def _chat_api(
        self,
        api_contents: List[Dict],
        stream: bool,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        topk: Optional[int] = None,
        retries: int = 2
     ) -> AsyncGenerator[Union[str, Dict], None]:
        """核心 API 调用逻辑"""
        
        body = {"contents": api_contents}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config = {}
        if max_output_tokens is not None and max_output_tokens > 0:
            generation_config["maxOutputTokens"] = max_output_tokens
        if topp is not None:
            generation_config["topP"] = max(0.0, min(1.0, topp))
        if temperature is not None:
            generation_config["temperature"] = max(0.0, min(2.0, temperature))
        if topk is not None and topk > 0:
             generation_config["topK"] = topk
        if thinking_budget is not None:
             generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        if generation_config:
            body["generationConfig"] = generation_config

        endpoint = f"/v1beta/models/{self.model}:{'streamGenerateContent' if stream else 'generateContent'}"
        logger.info(f"请求端点: {self.baseurl}{endpoint}")
        logger.debug(f"请求体: {json.dumps(body, ensure_ascii=False, indent=2)}")

        model_message_parts = []

        for attempt in range(retries):
             try:
                if stream:
                     logger.info("发起 Stream API 请求...")
                     full_text = ""
                     all_thoughts = []
                     async with self.client.stream("POST", endpoint, json=body, params={'alt': 'sse', 'key': self.apikey}) as response:
                          response.raise_for_status()
                          async for line in response.aiter_lines():
                               if not line.startswith("data: "):
                                    continue
                               data_str = line[len("data: "):].strip()
                               if not data_str: continue
                               try:
                                    chunk = json.loads(data_str)
                                    for candidate in chunk.get("candidates", []):
                                         for part in candidate.get("content", {}).get("parts", []):
                                              if "text" in part:
                                                  full_text += part["text"]
                                                  yield part["text"]
                                              if "thoughts" in part: # Handle thoughts
                                                   all_thoughts.append(part["thoughts"])
                                                   yield {"thoughts": part["thoughts"]}
                               except json.JSONDecodeError:
                                     logger.warning(f"Stream JSON 解析失败: {data_str}")
                     if full_text:
                           model_message_parts.append({"text": full_text})
                     if all_thoughts:
                           model_message_parts.append({"thoughts": all_thoughts})
                     logger.info("Stream API 请求完成。")
                     break # Success

                else: # Non-stream
                    logger.info(f"发起 Non-Stream API 请求 (尝试 {attempt+1}/{retries})...")
                    response = await self.client.post(endpoint, json=body, params={'key': self.apikey})
                    response.raise_for_status()
                    result = response.json()
                    if not result.get("candidates"):
                         logger.error(f"API 返回无 candidates: {result}")
                         prompt_feedback = result.get("promptFeedback", {})
                         if prompt_feedback:
                              logger.error(f"Prompt Feedback: {prompt_feedback}")
                         raise RuntimeError(f"API 返回无 candidates. Prompt Feedback: {prompt_feedback}")

                    candidate = result["candidates"][0]
                    content_parts = candidate.get("content", {}).get("parts", [])
                    model_message_parts.extend(content_parts) # Add all parts to history
                    
                    thoughts = [part["thoughts"] for part in content_parts if "thoughts" in part]
                    text = "".join(part["text"] for part in content_parts if "text" in part)
                   
                    if thoughts: # Yield dict if thoughts exist
                         yield {"thoughts": thoughts, "text": text}
                    elif text:
                         yield text
                    logger.info("Non-Stream API 请求完成。")
                    break # Success
            
             except (httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError, RuntimeError) as e:
                  err_content = ""
                  if isinstance(e, httpx.HTTPStatusError):
                       try: err_content = e.response.text 
                       except: pass
                  logger.error(f"API 调用失败 (尝试 {attempt+1}/{retries}): {type(e).__name__} - {str(e)} - {err_content}")
                  if attempt == retries - 1:
                      raise RuntimeError(f"API 调用在 {retries} 次重试后失败: {type(e).__name__} - {str(e)}") from e
                  await asyncio.sleep(1.5 ** attempt)
        
        # Add model response to history
        if model_message_parts:
            api_contents.append({"role": "model", "parts": model_message_parts})


     async def chat(
        self,
        messages: List[Dict[str, any]],
        stream: bool = False,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        topk: Optional[int] = None,
        retries: int = 2
    ) -> AsyncGenerator[Union[str, Dict], None]:

        api_contents = []
        for msg in messages:
             role = msg.get("role")
             if role == "assistant": role = "model"
             if role not in ["user", "model"]: 
                   logger.warning(f"跳过 Gemini 不支持的角色: {role}")
                   continue
             parts = msg.get("parts", [])
             if not isinstance(parts, list):
                  parts = [{"text": str(parts)}]
             api_contents.append({"role": role, "parts": parts})

        try:
             async for part in self._chat_api(
                api_contents, 
                stream, 
                max_output_tokens,
                system_instruction, 
                topp, 
                temperature, 
                thinking_budget,
                topk,
                retries
             ):
                yield part
        finally:
             messages.clear()
             for content in api_contents:
                  role = content.get("role")
                  if role == "model": role = "assistant"
                  messages.append({"role": role, "parts": content.get("parts", [])})

     async def close_client(self):
         if self.client and not self.client.is_closed:
            logger.info("Closing httpx client for Gemini...")
            await self.client.aclose()
            
API_INSTANCE_TYPE = "GEMINI_API_INSTANCE"
CONTENT_ITEM_TYPE = "GEMINI_CONTENT_ITEM"
HISTORY_TYPE = "STRING"

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        return asyncio.run(coro) 
    except RuntimeError:
        return asyncio.run(coro)

class GeminiApiLoaderNode:
    def __init__(self):
        self.cached_instance : Optional[GeminiAPI] = None
        self.cached_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "YOUR_GEMINI_KEY", "multiline": False}),
                "model": ("STRING", {"default": "gemini-2.0-flash-001", "multiline": False}),
                 "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com", "multiline": False}),
            },
             "optional": {
                 "proxy_http": ("STRING", {"default": "", "multiline": False, "placeholder": "http://127.0.0.1:7890"}),
                 "proxy_https": ("STRING", {"default": "", "multiline": False, "placeholder": "http://127.0.0.1:7890"}),
                 "timeout": ("FLOAT", {"default": 180.0, "min": 10.0, "max": 600.0, "step": 1.0}),
             }
        }
    RETURN_TYPES = (API_INSTANCE_TYPE,)
    RETURN_NAMES = ("api_instance",)
    FUNCTION = "load_api"
    CATEGORY = "Gemini API"

    def load_api(self, api_key: str, model: str, base_url:str, timeout: float, proxy_http: str = "", proxy_https: str = ""):
        proxies = {}
        if proxy_http: proxies["http://"] = proxy_http
        if proxy_https: proxies["https://"] = proxy_https
        if not proxies: proxies = None
        
        config_str = f"{api_key}{model}{base_url}{proxy_http}{proxy_https}{timeout}"
        current_hash = hash(config_str)

        if self.cached_instance and self.cached_config_hash == current_hash:
             logger.info("使用缓存的 Gemini API 实例")
             return (self.cached_instance,)

        if self.cached_instance:
              logger.info("配置改变，关闭旧的 Gemini API 客户端...")
              try: run_async(self.cached_instance.close_client())
              except Exception as e: logger.warning(f"关闭旧客户端失败: {e}")
              self.cached_instance = None

        logger.info("创建新的 Gemini API 实例...")
        try:
            instance = GeminiAPI(apikey=api_key.strip(), baseurl=base_url.strip(), model=model.strip(), proxies=proxies, timeout=timeout)
            self.cached_instance = instance
            self.cached_config_hash = current_hash
            return (instance,)
        except Exception as e:
             logger.error(f"创建 GeminiAPI 实例失败: {e}")
             raise

class GeminiImageEncoderNode:
    """将 ComfyUI Image 编码为 Gemini inlineData 块"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                 "format": (["png", "jpeg", "webp", "heic", "heif"], {"default": "jpeg"}), # Gemini supported
                 "quality": ("INT", {"default": 85, "min": 10, "max": 100, "step": 1}),
            },
        }
    RETURN_TYPES = (CONTENT_ITEM_TYPE, "STRING", )
    RETURN_NAMES = ("content_item", "base64_data",)
    FUNCTION = "encode_image"
    CATEGORY = "Gemini API/Content"

    def encode_image(self, image: 'torch.Tensor', format: str, quality: int):
         if image is None: raise ValueError("输入图像不能为空")
         logger.info(f"接收到 {image.shape[0]} 张图像，编码为 {format} (quality={quality})")
         try:
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if format.lower() == 'jpeg' and img.mode == 'RGBA': img = img.convert('RGB')

            buffer = io.BytesIO()
            save_params = {}
            if format.lower() in ['jpeg', 'webp']: save_params['quality'] = quality
            img.save(buffer, format=format.upper() if format.lower() != 'heic' else 'HEIF', **save_params)
            
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = f"image/{format.lower()}"
            # 构造 Gemini parts 所需的结构
            content_item = {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_data
                }
            }
            logger.info(f"图像编码成功 -> inlineData")
            return (content_item, base64_data)
         except Exception as e:
              logger.error(f"图像编码失败: {e}")
              raise

class GeminiFileUploaderNode:
    """上传文件到 Gemini File API 并返回 fileData 块"""
    @classmethod
    def INPUT_TYPES(cls):
         input_dir = folder_paths.get_input_directory()
         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith(".")]
         return {
            "required": {
                 "api_instance": (API_INSTANCE_TYPE, ),
                 "file_selector": (sorted(files) if files else ["No files in input dir"], ),
                 "use_absolute_path": ("BOOLEAN", {"default": False}),
                 "absolute_path_override": ("STRING", {"default": "/path/to/your/file.pdf", "multiline": False}),
                 "display_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional display name"}),
            },
        }
    RETURN_TYPES = (CONTENT_ITEM_TYPE, "STRING",)
    RETURN_NAMES = ("content_item", "file_uri", )
    FUNCTION = "upload_file"
    CATEGORY = "Gemini API/Content"

    def upload_file(self, api_instance: GeminiAPI, file_selector: str, use_absolute_path: bool, absolute_path_override: str, display_name: str):
        if not api_instance: raise ValueError("API 实例未连接")
        file_path = ""
        if use_absolute_path and absolute_path_override: file_path = absolute_path_override
        else: file_path = folder_paths.get_annotated_filepath(file_selector)
        
        if not file_path or not os.path.exists(file_path):
             raise FileNotFoundError(f"文件路径无效或文件不存在: {file_path}")
        logger.info(f"准备上传文件: {file_path}")

        async def do_upload():
             # api.upload_file 已经返回了 {"fileData": ...} 结构
            return await api_instance.upload_file(file_path, display_name if display_name else None)
        try:
            # result = {"fileData": {"mimeType": mime_type, "fileUri": file_uri} }
            result = run_async(do_upload())
            file_uri = result.get("fileData", {}).get("fileUri", "ERROR_NO_URI")
            logger.info(f"文件上传完成 -> fileData, URI: {file_uri}")
            return (result, file_uri)
        except Exception as e:
             logger.error(f"文件上传节点错误: {e}")
             raise

class GeminiTextBlockNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True}),}}
    RETURN_TYPES = (CONTENT_ITEM_TYPE,)
    RETURN_NAMES = ("content_item",)
    FUNCTION = "create_text_block"
    CATEGORY = "Gemini API/Content"
    def create_text_block(self, text: str):
         return ({"text": text},) # Gemini parts text format

class GeminiChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_instance": (API_INSTANCE_TYPE, ),
                "user_prompt": ("STRING", {"default": "你好", "multiline": True}),
                 "stream": ("BOOLEAN", {"default": False}),
                 "filter_thoughts": ("BOOLEAN", {"default": True, "label_on": "Filter THOUGHTS", "label_off": "Keep THOUGHTS"}),
            },
            "optional": {
                 "system_prompt": ("STRING", {"default": "", "multiline": True}),
                 "history_json_in": (HISTORY_TYPE, {"default": "[]", "multiline": True, "dynamicPort": True, "tooltip": "可以把history_json_out连接到这里来实现多轮对话"}),
                 "content_part_1": (CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                 "content_part_2": (CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                 "content_part_3": (CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                 "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 100000, "step": 1}),
                 "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                 "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "description": "0 or none to disable"}),
                 "thinking_budget": ("INT", {"default": -1, "min": -1, "max": 24576, "step": 1, "description":"-1: disable, 0: no thinking, >0: budget (1.5 models only)"}),
                 "retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "should_change": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("STRING", HISTORY_TYPE,)
    RETURN_NAMES = ("response_text", "history_json_out",)
    FUNCTION = "chat"
    CATEGORY = "Gemini API"

    def chat(self, api_instance: GeminiAPI, user_prompt: str, stream: bool, filter_thoughts: bool,
             system_prompt: str = "", history_json_in: str = "[]", 
             content_part_1: Optional[Dict] = None, content_part_2: Optional[Dict] = None, content_part_3: Optional[Dict] = None,
             max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 0, 
             thinking_budget: int = -1, retries: int = 1,should_change: bool = False,
             ):
        if not api_instance: raise ValueError("API 实例未连接")
        try:
            messages = json.loads(history_json_in or "[]")
            if not isinstance(messages, list): messages = []
        except json.JSONDecodeError:
            messages = []
            logger.warning("无法解析历史记录JSON，已重置。")

        user_parts = []
        if user_prompt: user_parts.append({"text": user_prompt})
        possible_parts = [content_part_1, content_part_2, content_part_3]
        for part in possible_parts:
             if part and isinstance(part, dict):
                  user_parts.append(part)
        
        if not user_parts:
             logger.warning("用户提示词和内容块均为空。")
             return ("", json.dumps(messages, ensure_ascii=False, indent=2))

        messages.append({"role": "user", "parts": user_parts})

        async def run_chat_task(msg_list: List[Dict]):
             final_text_parts = []
             gen = api_instance.chat(
                  messages=msg_list,
                  stream=stream,
                  system_instruction=system_prompt if system_prompt else None,
                  max_output_tokens=max_tokens,
                  temperature=temperature,
                  topp=top_p,
                  topk=top_k if top_k > 0 else None,
                  thinking_budget= thinking_budget if thinking_budget >=0 else None,
                  retries=retries
             )
             try:
                async for part in gen:
                    if isinstance(part, dict):
                        thoughts = part.get("thoughts")
                        text = part.get("text", "")
                        if thoughts:
                             thoughts_str = json.dumps(thoughts, ensure_ascii=False)
                             logger.info(f"THOUGHTS: {thoughts_str}")
                             if not filter_thoughts:
                                  final_text_parts.append(f"\n[THOUGHTS]: {thoughts_str}\n")
                        if text:
                            final_text_parts.append(text)
                    elif isinstance(part, str):
                         final_text_parts.append(part)
             except Exception as e:
                  logger.error(f"Chat API 异步生成器错误: {e}")
                  raise 
             return "".join(final_text_parts)

        try:
            logger.info(f"开始 Gemini 聊天请求 (Stream={stream})...")
            final_text = run_async(run_chat_task(messages)) 
            logger.info("Gemini 聊天请求结束。")
        except Exception as e:
             final_text = f"[NODE ERROR]: {type(e).__name__} - {str(e)}"
             logger.error(f"节点执行 Gemini 聊天任务失败: {e}")
             messages.append({"role": "assistant", "parts": [{"text": final_text}]})
            
        try:
            history_json_out = json.dumps(messages, ensure_ascii=False, indent=2)
        except TypeError as e:
             logger.error(f"序列化历史记录失败: {e}")
             history_json_out = "[]"
        return (final_text, history_json_out)

    @classmethod
    def IS_CHANGED(s, should_change=False, *args, **kwargs):
        if should_change:
            return float("NaN")
        else:
            return False
