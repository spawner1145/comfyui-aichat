import httpx
import json
import mimetypes
import base64
import os
import uuid
from typing import Generator, Dict, List, Optional, Union, Any
import logging
import io
import time

#default_system = "你是一个善于写ai画图提示词的ai助手，擅长润色提示词，描述图片，并且可以把我输入的文本和输入的图片的特征结合起来润色，不要有多余的话，直接输出描述词，结合自然语言和danbooru tags详细描述，注意千万不要忘记自然语言"
default_system = """将danboorutag标签和图片中的特征结合(如果有)转化成自然语言，你可以将这些标签和图片(如果有)作为参考，但不要完全依赖它们，因为可能存在错误的标签。优先使用你自己的观察，并使用更合适的同义词进行描述。- 注意事项：
     - 确保场景描述的准确性。
     - 如果场景不符合上述规则或无法分析，请勿描述。保持描述的准确性。
     - 使用清晰简洁的语言进行描述。避免使用复杂或难以理解的语言。
     - 描述时尽量客观。避免使用高度主观的语言。
     - 如果场景包含NSFW或敏感内容，请勿回避。直接描述露骨内容。
     - 如果场景包含暴力、血腥、恐怖、恶心或其他敏感内容，请勿回避。直接描述。
     - 使用**英文**进行描述。
     - 想象你是一名用户试图描绘这个场景。自然地结合内容，避免结构化的描述。
     - 只返回描述。不要包含额外内容或分析。
     - 你的任务是描述图像中的每个方面、对象和互动，使得盲人在听到描述时能够在脑海中完美捕捉到图像。

     - ！！！最重要的一点，#符号后面的tag是角色名，@符号后面的是画师名，一定要提到这两个。#和@是极其重要的特殊标签，不能删除,要将二者放在开头，作为固定开头，例如：Characters: #hoshimi miyabi. Drawn by @quan \(kurisu tina\)+xxxx（自然语言部分）。

特殊标签很重要
记住了吗？"""
default_system = "You are a Prompt optimizer designed to rewrite user inputs into high-quality Prompts that are more complete and expressive while preserving the original meaning.\nTask Requirements:\n1. For overly brief user inputs, reasonably infer and add details to enhance the visual completeness without altering the core content;\n2. Refine descriptions of subject characteristics, visual style, spatial relationships, and shot composition;\n3. If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated;\n4. Match the Prompt to a precise, niche style aligned with the user’s intent. If unspecified, choose the most appropriate style (e.g., realistic photography style);\n5. Please ensure that the Rewritten Prompt is less than 200 words.\nRewritten Prompt Examples:\n1. Dunhuang mural art style: Chinese animated illustration, masterwork. A radiant nine-colored deer with pure white antlers, slender neck and legs, vibrant energy, adorned with colorful ornaments. Divine flying apsaras aura, ethereal grace, elegant form. Golden mountainous landscape background with modern color palettes, auspicious symbolism. Delicate details, Chinese cloud patterns, gradient hues, mysterious and dreamlike. Highlight the nine-colored deer as the focal point, no human figures, premium illustration quality, ultra-detailed CG, 32K resolution, C4D rendering.\n2. Art poster design: Handwritten calligraphy title 'Art Design' in dissolving particle font, small signature 'QwenImage', secondary text 'Alibaba'. Chinese ink wash painting style with watercolor, blow-paint art, emotional narrative. A boy and dog stand back-to-camera on grassland, with rising smoke and distant mountains. Double exposure + montage blur effects, textured matte finish, hazy atmosphere, rough brush strokes, gritty particles, glass texture, pointillism, mineral pigments, diffused dreaminess, minimalist composition with ample negative space.\n3. Black-haired Chinese adult male, portrait above the collar. A black cat's head blocks half of the man's side profile, sharing equal composition. Shallow green jungle background. Graffiti style, clean minimalism, thick strokes. Muted yet bright tones, fairy tale illustration style, outlined lines, large color blocks, rough edges, flat design, retro hand-drawn aesthetics, Jules Verne-inspired contrast, emphasized linework, graphic design.\n4. Fashion photo of four young models showing phone lanyards. Diverse poses: two facing camera smiling, two side-view conversing. Casual light-colored outfits contrast with vibrant lanyards. Minimalist white/grey background. Focus on upper bodies highlighting lanyard details.\n5. Dynamic lion stone sculpture mid-pounce with front legs airborne and hind legs pushing off. Smooth lines and defined muscles show power. Faded ancient courtyard background with trees and stone steps. Weathered surface gives antique look. Documentary photography style with fine details."

try:
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    import folder_paths
    print("Loaded ComfyUI modules successfully.")
except ImportError:
    print("Warning: Could not import ComfyUI modules (torch, numpy, PIL, folder_paths). Nodes will not work outside ComfyUI.")
    torch = None
    Image = None
    folder_paths = None

logger = logging.getLogger('OpenAI_ComfyUI')
logger.setLevel(logging.INFO)
from openai import OpenAI

class OpenAIAPI:
    def __init__(
        self,
        apikey: str,
        baseurl: str = "https://api-inference.modelscope.cn/v1/",
        model: str = "deepseek-ai/DeepSeek-R1",
        proxies: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,
    ):
        if not apikey:
            raise ValueError("API Key 不能为空")
        self.apikey = apikey
        self.baseurl = baseurl if baseurl.endswith('/') else baseurl + '/'
        self.model = model
        http_client = httpx.Client(proxies=proxies, timeout=timeout) if proxies else httpx.Client(timeout=timeout)
        self.client = OpenAI(
            api_key=apikey,
            base_url=self.baseurl,
            http_client=http_client
        )
        logger.info(f"OpenAIAPI Client Initialized: model={self.model}, base_url={self.baseurl}")

    def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, Union[str, None]]:
        if not os.path.exists(file_path):
            logger.error(f"文件 {file_path} 不存在")
            raise FileNotFoundError(f"文件上传失败: 路径 {file_path} 不存在")
            
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            logger.error(f"获取文件 {file_path} 大小失败: {e}")
            raise

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
            logger.warning(f"无法检测文件 {file_path} 的 MIME 类型，使用默认值: {mime_type}")

        try:
            logger.info(f"开始上传文件: {file_path} ({mime_type})")
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_tuple = (display_name or os.path.basename(file_path), file_content, mime_type)
                file_obj = self.client.files.create(
                    file=file_tuple,
                    purpose="user_data"
                )
                final_file_obj = None
                if hasattr(file_obj, 'id') and response_obj.id:
                    logger.info("API 响应为标准格式，直接使用。")
                    final_file_obj = file_obj
                else:
                    logger.info("响应非标准格式，开始搜索嵌套的文件对象...")
                    file_dict = file_obj.model_dump()
                    
                    found_nested_dict = None
                    for key, value in file_dict.items():
                        if isinstance(value, dict) and 'id' in value:
                            logger.info(f"在键 '{key}' 中找到疑似嵌套的文件对象。")
                            found_nested_dict = value
                            break

                    if found_nested_dict:
                        logger.info("从嵌套的字典中成功重建标准文件对象")
                        final_file_obj = FileObject(**found_nested_dict)

                if final_file_obj and hasattr(final_file_obj, 'id'):
                    file_id = final_file_obj.id
                    logger.info(f"文件 {file_path} 上传并处理成功, ID: {file_id}")
                    return {"input_file": {"file_id": file_id}, "error": None}
                else:
                    error_message = f"无法在API响应中找到有效的文件对象。响应内容: {file_obj.model_dump_json()}"
                    logger.error(error_message)
                    raise ValueError(error_message)

        except Exception as e:
            logger.error(f"文件 {file_path} 上传失败: {type(e).__name__} - {str(e)}")
            raise RuntimeError(f"文件 {file_path} 上传失败: {type(e).__name__} - {str(e)}") from e

    def _chat_api(
        self,
        messages: List[Dict],
        stream: bool,
        max_output_tokens: Optional[int] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        retries: int = 2
    ) -> Generator[str, None, None]:
        api_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")

            if role not in ["user", "assistant", "system", "tool"]:
                logger.warning(f"跳过无效角色消息: {role}")
                continue

            api_msg = {"role": role }
            if role == "tool":
                api_msg["content"] = content if isinstance(content, str) else json.dumps(content)
                if tool_call_id:
                    api_msg["tool_call_id"] = tool_call_id
                if name:
                    api_msg["name"] = name
                api_messages.append(api_msg)
                continue

            if role == "system":
                if isinstance(content, str):
                    api_msg["content"] = content
                elif isinstance(content, list):
                    all_text = " ".join([p.get("text", "") for p in content if p.get("type") == "text"])
                    api_msg["content"] = all_text or " "
                else:
                    api_msg["content"] = str(content)
                api_messages.append(api_msg)
                continue

            api_content = []
            if isinstance(content, str):
                api_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict): continue

                    if part.get("type") == "text" and "text" in part:
                        api_content.append(part)
                    elif "input_file" in part and "file_id" in part["input_file"]:
                        api_content.append({
                            "type": "file",
                            "file_id": part["input_file"]["file_id"]
                        })
                    elif "input_image" in part and "image_url" in part["input_image"]:
                        api_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": part["input_image"]["image_url"],
                                "detail": part["input_image"].get("detail", "auto")
                            }
                        })
                    elif part.get("type") in ["image_url", "file", "input_file", "file_id"]:
                        api_content.append(part)
                    else:
                        logger.warning(f"跳过无法识别的消息内容块: {part}")
            else:
                api_content = [{"type": "text", "text": str(content)}]
            
            if api_content:
                api_msg["content"] = api_content if len(api_content) > 1 or not (api_content[0].get("type") == "text") else api_content[0].get("text", "")
                if role == "assistant" and isinstance(api_content, list) and len(api_content) == 1 and api_content[0].get("type") == "text":
                    api_msg["content"] = api_content[0].get("text", "")

                if "tool_calls" in msg and role == "assistant":
                    api_msg["tool_calls"] = msg["tool_calls"]
                    if not isinstance(api_msg.get("content"), str):
                        api_msg["content"] = None 

                api_messages.append(api_msg)
            elif role == "assistant" and "tool_calls" in msg:
                api_msg["content"] = None
                api_msg["tool_calls"] = msg["tool_calls"]
                api_messages.append(api_msg)

        logger.debug(f"发送给 API 的 Messages: {json.dumps(api_messages, ensure_ascii=False, indent=2)}")

        request_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream,
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            request_params["max_tokens"] = max_output_tokens
        if topp is not None:
            request_params["top_p"] = max(0.0, min(1.0, topp))
        if temperature is not None:
            request_params["temperature"] = max(0.0, min(2.0, temperature))

        request_params = {k: v for k, v in request_params.items() if v is not None}

        assistant_content_text = ""
        full_reasoning = []

        for attempt in range(retries):
            try:
                if stream:
                    logger.info("发起 Stream API 请求...")
                    stream_resp = self.client.chat.completions.create(**request_params)
                    for chunk in stream_resp:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            full_reasoning.append(delta.reasoning_content)
                            yield f"REASONING: {delta.reasoning_content}\n"
                        if delta and delta.content:
                            assistant_content_text += delta.content
                            yield delta.content
                    if assistant_content_text or full_reasoning:
                        messages.append({
                            "role": "assistant", 
                            "content": [{"type": "text", "text": assistant_content_text}],
                        })
                    logger.info("Stream API 请求完成。")
                    return

                else:
                    logger.info(f"发起 Non-Stream API 请求 (尝试 {attempt+1}/{retries})...")
                    response = self.client.chat.completions.create(**request_params)
                    if not response.choices:
                        raise RuntimeError("API 返回空 choices")
                    
                    choice = response.choices[0]
                    message = choice.message
                        
                    if hasattr(message, 'reasoning_content') and message.reasoning_content:
                        full_reasoning.append(message.reasoning_content)
                        yield f"REASONING: {message.reasoning_content}\n"

                    content_to_yield = message.content or ""
                    assistant_content_text = content_to_yield

                    assistant_message = {
                        "role": "assistant",
                        "content": content_to_yield
                    }
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        assistant_message["tool_calls"] = [tc.dict() for tc in message.tool_calls]
                        assistant_message["content"] = message.content

                    messages.append(assistant_message)
                    logger.info("Non-Stream API 请求完成。")
                    yield content_to_yield
                    return

            except Exception as e:
                logger.error(f"API 调用失败 (尝试 {attempt+1}/{retries}): {type(e).__name__} - {str(e)}")
                if attempt == retries - 1:
                    raise RuntimeError(f"API 调用在 {retries} 次重试后失败: {type(e).__name__} - {str(e)}") from e
                time.sleep(1.5 ** attempt)

    def chat(
        self,
        messages: List[Dict[str, any]],
        stream: bool = False,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        retries: int = 2
    ) -> Generator[str, None, None]:
        current_messages = list(messages)

        if system_instruction:
            system_found = False
            for i, message in enumerate(current_messages):
                if message.get("role") == "system":
                    current_messages[i] = {"role": "system", "content": system_instruction}
                    system_found = True
                    break
            if not system_found:
                current_messages.insert(0, {"role": "system", "content": system_instruction})

        full_response_parts = []
        try:
            for part in self._chat_api(
                current_messages,
                stream, 
                max_output_tokens,
                topp, 
                temperature,
                retries
            ):
                full_response_parts.append(part)
                yield part
        finally:
            messages.clear()
            messages.extend(current_messages)

    def close_client(self):
        if self.client and hasattr(self.client, 'close'):
            logger.info("Closing httpx client...")
            self.client.close()

API_INSTANCE_TYPE = "OPENAI_API_INSTANCE"
CONTENT_ITEM_TYPE = "OAI_CONTENT_ITEM"
HISTORY_TYPE = "STRING"

class OpenAIApiLoaderNode:
    def __init__(self):
        self.cached_instance : Optional[OpenAIAPI] = None
        self.cached_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "sk-xxxx", "multiline": False}),
                "model": ("STRING", {"default": "deepseek-ai/DeepSeek-R1", "multiline": False}),
                "base_url": ("STRING", {"default": "https://api-inference.modelscope.cn/v1/", "multiline": False}),
            },
            "optional": {
                "proxy_http": ("STRING", {"default": "", "multiline": False, "placeholder": "http://127.0.0.1:7890"}),
                "proxy_https": ("STRING", {"default": "", "multiline": False, "placeholder": "http://127.0.0.1:7890"}),
                "timeout": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 600.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = (API_INSTANCE_TYPE,)
    RETURN_NAMES = ("api_instance",)
    FUNCTION = "load_api"
    CATEGORY = "OpenAI API"

    def load_api(self, api_key: str, model: str, base_url:str, timeout: float, proxy_http: str = "", proxy_https: str = ""):
        proxies = {}
        if proxy_http: proxies["http://"] = proxy_http
        if proxy_https: proxies["https://"] = proxy_https
        if not proxies: proxies = None

        config_str = f"{api_key}{model}{base_url}{proxy_http}{proxy_https}{timeout}"
        current_hash = hash(config_str)

        if self.cached_instance and self.cached_config_hash == current_hash:
            logger.info("使用缓存的 OpenAI API 实例")
            return (self.cached_instance,)

        if self.cached_instance:
            logger.info("配置改变，关闭旧的 OpenAI API 客户端...")
            try:
                self.cached_instance.close_client()
            except Exception as e:
                logger.warning(f"关闭旧客户端失败: {e}")
            self.cached_instance = None

        logger.info("创建新的 OpenAI API 实例...")
        try:
            instance = OpenAIAPI(
                apikey=api_key.strip(),
                baseurl=base_url.strip(),
                model=model.strip(),
                proxies=proxies,
                timeout=timeout,
            )
            self.cached_instance = instance
            self.cached_config_hash = current_hash
            return (instance,)
        except Exception as e:
            logger.error(f"创建 OpenAIAPI 实例失败: {e}")
            raise

class OpenAIImageEncoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail": (["auto", "low", "high"], {"default": "auto"}),
                "format": (["png", "jpeg", "webp"], {"default": "jpeg"}),
                "quality": ("INT", {"default": 85, "min": 10, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = (CONTENT_ITEM_TYPE, "STRING", )
    RETURN_NAMES = ("content_item", "base64_string",)
    FUNCTION = "encode_image"
    CATEGORY = "OpenAI API/Content"
    OUTPUT_IS_LIST = (False, False,)

    def encode_image(self, image: 'torch.Tensor', detail: str, format: str, quality: int):
        if image is None:
            raise ValueError("输入图像不能为空")
        
        logger.info(f"接收到 {image.shape[0]} 张图像，编码为 {format} (quality={quality}, detail={detail})")

        try:
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            if format.lower() == 'jpeg' and img.mode == 'RGBA':
                img = img.convert('RGB')

            buffer = io.BytesIO()
            save_params = {}
            if format.lower() in ['jpeg', 'webp']:
                save_params['quality'] = quality
            img.save(buffer, format=format.upper(), **save_params)
            
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = f"image/{format.lower()}"
            data_url = f"data:{mime_type};base64,{base64_data}"
            
            content_item = {
                "input_image": {
                    "image_url": data_url,
                    "detail": detail
                }
            }
            logger.info(f"图像编码成功")
            return (content_item, data_url)

        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise

class OpenAIFileUploaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".DS_Store"]
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
    RETURN_NAMES = ("content_item", "file_id", )
    FUNCTION = "upload_file"
    CATEGORY = "OpenAI API/Content"

    def upload_file(self, api_instance: OpenAIAPI, file_selector: str, use_absolute_path: bool, absolute_path_override: str, display_name: str):
        if not api_instance:
            raise ValueError("API 实例未连接")
        
        file_path = ""
        if use_absolute_path and absolute_path_override:
            file_path = absolute_path_override
            logger.info(f"使用绝对路径: {file_path}")
        else:
            file_path = folder_paths.get_annotated_filepath(file_selector)
            logger.info(f"使用 ComfyUI Input 目录路径: {file_path}")

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"文件路径无效或文件不存在: {file_path}")

        try:
            result = api_instance.upload_file(file_path, display_name if display_name else None)
            if result and "input_file" in result and result["input_file"] and not result.get("error"):
                file_id = result["input_file"].get("file_id", "ERROR_NO_ID")
                logger.info(f"文件上传完成，file_id: {file_id}")
                return (result, file_id)
            else:
                error_msg = result.get("error", "未知上传错误")
                logger.error(f"文件上传节点错误: {error_msg}")
                raise RuntimeError(f"文件上传失败: {error_msg}")
        except Exception as e:
            logger.error(f"文件上传执行错误: {e}")
            raise

class OpenAITextBlockNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }
    RETURN_TYPES = (CONTENT_ITEM_TYPE,)
    RETURN_NAMES = ("content_item",)
    FUNCTION = "create_text_block"
    CATEGORY = "OpenAI API/Content"

    def create_text_block(self, text: str):
        return ({"type": "text", "text": text},)

class OpenAIChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_instance": (API_INSTANCE_TYPE, ),
                "user_prompt": ("STRING", {"default": "你好", "multiline": True}),
                "stream": ("BOOLEAN", {"default": False}),
                "filter_reasoning": ("BOOLEAN", {"default": True, "label_on": "Filter REASONING:", "label_off": "Keep REASONING:"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": default_system, "multiline": True}),
                "history_json_in": (HISTORY_TYPE, {"default": "[]", "multiline": True, "dynamicPort": True, "tooltip": "可以把history_json_out连接到这里来实现多轮对话"}),
                "content_part_1": (CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                "content_part_2": (CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                "content_part_3": (CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32000, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "should_change": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", HISTORY_TYPE,)
    RETURN_NAMES = ("response_text", "history_json_out",)
    FUNCTION = "chat"
    CATEGORY = "OpenAI API"

    def chat(self, api_instance: OpenAIAPI, user_prompt: str, stream: bool, filter_reasoning: bool,
             system_prompt: str = "", history_json_in: str = "[]", 
             content_part_1: Optional[Dict] = None,
             content_part_2: Optional[Dict] = None,
             content_part_3: Optional[Dict] = None,
             max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.95, retries: int = 1, should_change: bool = False,
             ):
        if not api_instance:
            raise ValueError("API 实例未连接")

        try:
            messages = json.loads(history_json_in or "[]")
            if not isinstance(messages, list):
                messages = []
                logger.warning("历史记录JSON格式错误，已重置为空列表。")
        except json.JSONDecodeError:
            messages = []
            logger.warning("无法解析历史记录JSON，已重置为空列表。")

        user_content = []
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        
        possible_parts = [content_part_1, content_part_2, content_part_3]
        for part in possible_parts:
            if part and isinstance(part, dict):
                if "input_image" in part or "input_file" in part or "type" in part:
                    user_content.append(part)
                else:
                    logger.warning(f"跳过无效的内容块输入: {part}")

        final_user_content = user_content

        if not final_user_content:
            logger.warning("用户提示词和内容块均为空，跳过API调用。")
            return ("", json.dumps(messages, ensure_ascii=False, indent=2))
        
        messages.append({"role": "user", "content": final_user_content})

        full_parts = []
        final_text = ""
        try:
            logger.info(f"开始聊天请求 (Stream={stream})...")
            chat_generator = api_instance.chat(
                messages=messages,
                stream=stream,
                system_instruction=system_prompt if system_prompt else None,
                max_output_tokens=max_tokens,
                temperature=temperature,
                topp=top_p,
                retries=retries
            )
            
            for part in chat_generator:
                if filter_reasoning and part.startswith("REASONING:"):
                    logger.info(part.strip())
                    continue
                full_parts.append(part)
            
            final_text = "".join(full_parts)
            logger.info("聊天请求结束。")

        except Exception as e:
            final_text = f"[NODE ERROR]: {type(e).__name__} - {str(e)}"
            logger.error(f"节点执行聊天任务失败: {e}")
            messages.append({"role": "assistant", "content": final_text})

        try:
            history_json_out = json.dumps(messages, ensure_ascii=False, indent=2)
        except TypeError as e:
            logger.error(f"序列化历史记录失败: {e}. 历史: {messages}")
            history_json_out = json.dumps([{"role": "system", "content": f"History serialization error: {e}"}], ensure_ascii=False, indent=2)

        return (final_text, history_json_out)
    
    @classmethod
    def IS_CHANGED(s, should_change=False, *args, **kwargs):
        if should_change:
            return float("NaN")
        else:
            return False
