from .nodes import OpenAIApiLoaderNode, OpenAIImageEncoderNode, OpenAIFileUploaderNode, OpenAITextBlockNode, OpenAIChatNode, OpenAIContentConnector
from .gemini_nodes import GeminiApiLoaderNode, GeminiImageEncoderNode, GeminiFileUploaderNode, GeminiTextBlockNode, GeminiChatNode, GeminiContentConnector
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
