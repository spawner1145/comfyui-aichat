from .nodes import OpenAIApiLoaderNode, OpenAIImageEncoderNode, OpenAIFileUploaderNode, OpenAITextBlockNode, OpenAIChatNode
from .gemini_nodes import GeminiApiLoaderNode, GeminiImageEncoderNode, GeminiFileUploaderNode, GeminiTextBlockNode, GeminiChatNode
NODE_CLASS_MAPPINGS = {
    "OpenAIApiLoader_Zho": OpenAIApiLoaderNode,
    "OpenAIImageEncoder_Zho": OpenAIImageEncoderNode,
    "OpenAIFileUploader_Zho": OpenAIFileUploaderNode,
    "OpenAITextBlock_Zho": OpenAITextBlockNode,
    "OpenAIChat_Zho": OpenAIChatNode,
    "GeminiApiLoader_Zho": GeminiApiLoaderNode,
    "GeminiImageEncoder_Zho": GeminiImageEncoderNode,
    "GeminiFileUploader_Zho": GeminiFileUploaderNode,
    "GeminiTextBlock_Zho": GeminiTextBlockNode,
    "GeminiChat_Zho": GeminiChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIApiLoader_Zho": "OpenAI API 加载器",
    "OpenAIImageEncoder_Zho": "OpenAI 图像编码器",
     "OpenAIFileUploader_Zho": "OpenAI 文件上传器",
     "OpenAITextBlock_Zho": "OpenAI 文本块",
    "OpenAIChat_Zho": "OpenAI 聊天节点",
    "GeminiApiLoader_Zho": "Gemini API 加载器",
    "GeminiImageEncoder_Zho": "Gemini 图像编码器 (Inline)",
     "GeminiFileUploader_Zho": "Gemini 文件上传器 (File API)",
     "GeminiTextBlock_Zho": "Gemini 文本块",
    "GeminiChat_Zho": "Gemini 聊天节点",
}
