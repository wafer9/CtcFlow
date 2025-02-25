import tiktoken

from transformers import PreTrainedTokenizer

from .tokenization_qwen import QWenTokenizer, _load_tiktoken_bpe

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (
                     ENDOFTEXT,
                     IMSTART,
                     IMEND,
                 ) + EXTRAS
GAP = tuple((f"<|GAP_{i}|>" for i in range(85)))

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "it": "italian",
}


class QWenAudioTokenizer(QWenTokenizer, PreTrainedTokenizer):

    def __init__(
            self,
            vocab_file,
            errors="replace",
            audio_start_tag='<audio>',
            audio_end_tag='</audio>',
            **kwargs,
    ):
        PreTrainedTokenizer.__init__(self, **kwargs)
        self.eos_token = ENDOFTEXT
        self.audio_start_tag = audio_start_tag
        self.audio_end_tag = audio_end_tag
        self.audio_pad_tag = "[[[AUDIO:modality]]]"

        self.AUDIO_ST = (
            '[[[AUDIO:modality]]]',
            # Transcription Tag
            "<|startoftranscript|>",  # Transcription
            "<|startofanalysis|>",  # Analysis
            # Task Tag
            "<|translate|>",
            "<|transcribe|>",
            "<|caption|>",
            "<|keyword|>",
            # Language Tag
            "<|unknown|>",  # unknown language
            *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
            "<|zh_tr|>",  # tranditional Chinese
            # Timestamps Tag
            "<|notimestamps|>",
            "<|sil|>",
            "<|timestamps|>",
            *[f"<|{i * 0.01:.2f}|>" for i in range(3001)],  # timestamps 0.00-30.00
            # Output Instruction
            "<|caption_audiocaps|>",  # Audiocaps caption style
            "<|caption_clotho|>",  # Clotho caption style
            "<|audioset_ontology|>",  # Audioset ontology style
            "<|caption_plain|>",  # plain caption
            "<|itn|>",  # inversed text normalized
            "<|wo_itn|>",  # without inversed text normalized
            "<|startofentityvalue|>",
            "<|endofentityvalue|>",
            "<|startofentitytype|>",
            "<|endofentitytype|>",
            "<|named_entity_recognition|>",  # named entity recognition task
            "<|audio_grounding|>",
            "<|startofword|>",
            "<|endofword|>",
            "<|delim|>",  # delimiter of timestamps pair in audio grounding
            "<|emotion_recognition|>",  # emotion recognition
            "<|music_description|>",  # music description
            "<|note_analysis|>",  # note analysis
            "<|pitch|>",  # note analysis: pitch
            *[f"<|midi_pitch_{i}|>" for i in range(128)],  # midi pitch 0-127
            "<|velocity|>",  # note analysis: velocity
            *[f"<|midi_velocity_{i}|>" for i in range(128)],  # midi velocity 0-127
            "<|sonic|>",  # note analysis:  sonic
            "<|instrument|>",  # note analysis:  instrument
            "<|speaker_meta|>",  # meta information of speaker
            "<|song_meta|>",  # meta information of song
            "<|question|>",  # AQA: question
            "<|answer|>",  # AQA: answer
            "<|choice|>",  # AQA: answer choice
            "<|scene|>",  # scene recognition
            "<|event|>",  # sound event
            "<|vocal_classification|>",  # vocal classification
            "<|speech_understanding|>",  # speech language understanding
            "<|scenario|>",  # speech language understanding: scenario
            "<|action|>",  # speech language understanding: action
            "<|entities|>",  # speech language understanding: entities
            "<|speech_edit|>",  # speech edit
            audio_start_tag,
            audio_end_tag
        )

        self.errors = errors  # how to handle errors in decoding

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        self.special_tokens = {
            token: index
            for index, token in enumerate(
                SPECIAL_TOKENS + GAP + self.AUDIO_ST, start=len(self.mergeable_ranks)

            )
        }
        self.audio_start_id = self.special_tokens[self.audio_start_tag]
        self.audio_end_id = self.special_tokens[self.audio_end_tag]
        self.audio_pad_id = self.special_tokens[self.audio_pad_tag]
        self.timestamps = {
            'notimestamps': self.special_tokens['<|notimestamps|>'],
            'timestamps': self.special_tokens['<|timestamps|>'],
        }
        for i in range(3001):
            ts = f"<|{i * 0.01:.2f}|>"
            self.timestamps[ts] = self.special_tokens[ts]

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
                len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.special_tokens[ENDOFTEXT]
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]
