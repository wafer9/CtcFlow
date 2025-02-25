import tiktoken

from transformers import PreTrainedTokenizer

# from twinlakes.text.tokenization_qwen2 import Qwen2Tokenizer
from transformers import Qwen2TokenizerFast, AddedToken

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"


class Qwen2AudioTokenizer(Qwen2TokenizerFast):

    def __init__(
            self,
            vocab_file,
            merges_file,
            tokenizer_file=None,
            unk_token="<|endoftext|>",
            bos_token=None,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            **kwargs,
    ):
        super().__init__(vocab_file=vocab_file, 
                         merges_file=merges_file, 
                         tokenizer_file=tokenizer_file,
                         unk_token=unk_token,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         pad_token=pad_token,
                         **kwargs)

        self.eos_token = ENDOFTEXT
        self.audio_start_tag = "<audio>" # 152064
        self.audio_end_tag = "</audio>" # 152065
        self.audio_pad_tag = "[[[AUDIO:modality]]]" # 152066
        self.transcribe = "<|transcribe|>"


    def add_special_tokens(self, llm_vocab_size, spare=True):
        spare_num = llm_vocab_size - self.vocab_size - len(self.added_tokens_encoder)
        if spare:
            spare_tokens = list([f"<|GAP_{i}|>" for i in range(spare_num)])
        else:
            spare_tokens = []

        spare_tokens.extend([self.audio_start_tag,
                             self.audio_end_tag,
                             self.audio_pad_tag,
                             self.transcribe])

        self.add_tokens(spare_tokens, special_tokens=True)

        self.audio_start_id = self.encode(self.audio_start_tag)[0]
        self.audio_end_id = self.encode(self.audio_end_tag)[0]
        self.audio_pad_id = self.encode(self.audio_pad_tag)[0]