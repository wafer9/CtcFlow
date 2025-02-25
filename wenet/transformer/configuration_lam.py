from typing import Optional, Dict
from transformers import PretrainedConfig, AutoConfig
from wenet.transformer.configuration_qwen import QWenConfig


class LAMConfig(PretrainedConfig):
    model_type = "lam"  # 用于保存模型时自动保存对应的代码
    _auto_class = "AutoConfig"  # 用于保存模型时自动保存对应的代码
    is_composition = True  # TODO:
    keys_to_ignore_at_inference = [
        "past_key_values", "hidden_states", "attentions"
    ]  # 推理时不需要保存的输出，避免显存占用

    def __init__(
        self,
        audio_config: Optional[Dict] = {},
        text_config: Optional[Dict] = {},
        num_special_tokens_add: Optional[int] = 0,
        **kwargs,
    ):
        """
        num_special_tokens_add: 相较于原始 text model，添加的 special tokens 数量
        """
        super().__init__(**kwargs)

        audio_model_type = audio_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        if audio_model_type == "whisper_openai":
            from wenet.transformer.configuration_whisper_openai_encoder import WhisperConfig
            self.audio_config = WhisperConfig(**audio_config)
        else:
            self.audio_config = AutoConfig.for_model(audio_model_type,
                                                     **audio_config)

        if text_model_type == "qwen":
            # 自定义模型无法使用 `AutoConfig.for_model`
            self.text_config = QWenConfig(**text_config)
        else:
            # `transformers 包含的模型可以直接使用 model_type 匹配对应的 config 类及其默认值`
            self.text_config = AutoConfig.for_model(text_model_type,
                                                    **text_config)

        self.num_special_tokens_add = num_special_tokens_add
