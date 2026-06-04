from .nodes import OpenAIApiLoaderNode, OpenAIImageEncoderNode, OpenAIFileUploaderNode, OpenAITextBlockNode, OpenAIChatNode, OpenAIContentConnector
from .gemini_nodes import GeminiApiLoaderNode, GeminiImageEncoderNode, GeminiFileUploaderNode, GeminiTextBlockNode, GeminiChatNode, GeminiContentConnector

# 注册自定义前端目录（ComfyUI 会把 ./web 挂载到 /extensions/<module>/）
WEB_DIRECTORY = "./web"

# 注册「获取模型列表」后端路由（在 ComfyUI 中导入即注册；非 ComfyUI 环境下安全跳过）
try:
    from . import web_api  # noqa: F401
except Exception as _e:
    print(f"[aichat] 模型列表接口注册失败（不影响节点本身）: {_e}")

NODE_CLASS_MAPPINGS = {
    "OpenAIApiLoader": OpenAIApiLoaderNode,
    "OpenAIImageEncoder": OpenAIImageEncoderNode,
    "OpenAIFileUploader": OpenAIFileUploaderNode,
    "OpenAITextBlock": OpenAITextBlockNode,
    "OpenAIChat": OpenAIChatNode,
    "OpenAIContentConnector": OpenAIContentConnector,

    "GeminiApiLoader": GeminiApiLoaderNode,
    "GeminiImageEncoder": GeminiImageEncoderNode,
    "GeminiFileUploader": GeminiFileUploaderNode,
    "GeminiTextBlock": GeminiTextBlockNode,
    "GeminiChat": GeminiChatNode,
    "GeminiContentConnector": GeminiContentConnector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIApiLoader": "OpenAI API 加载器",
    "OpenAIImageEncoder": "OpenAI 图像编码器",
    "OpenAIFileUploader": "OpenAI 文件上传器",
    "OpenAITextBlock": "OpenAI 文本块",
    "OpenAIChat": "OpenAI 聊天节点",
    "OpenAIContentConnector": "OpenAI 内容块连接器",
    
    "GeminiApiLoader": "Gemini API 加载器",
    "GeminiImageEncoder": "Gemini 图像编码器 (Inline)",
    "GeminiFileUploader": "Gemini 文件上传器 (File API)",
    "GeminiTextBlock": "Gemini 文本块",
    "GeminiChat": "Gemini 聊天节点",
    "GeminiContentConnector": "Gemini 内容块连接器",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
