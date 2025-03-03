from typing import Optional, Tuple, Dict, List, Callable, Union
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, WhisperConfig, PreTrainedModel, AutoModel, AutoModelForCausalLM, ProcessorMixin, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateOutput
from transformers.models.whisper.modeling_whisper import _compute_mask_indices
from transformers.modeling_outputs import CausalLMOutputWithPast


from wenet.transformer.configuration_lam import LAMConfig

HistoryType = List[Tuple[str, str]]
from torch.nn.utils.rnn import pad_sequence
from wenet.transformer.m_adapter_block import TransformerAdaptorLayer
from wenet.transformer.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from fairseq import checkpoint_utils
from wenet.text.tokenization_qwen_audio import QWenAudioTokenizer
from fairseq.data import Dictionary

class LAMModel(PreTrainedModel):
    config_class = LAMConfig  # 用于保存模型时自动保存对应的代码
    _auto_class = "AutoModelForCausalLM"  # 用于保存模型时自动保存对应的代码

    _no_split_modules = [
        "WhisperEncoderLayer", "QWenBlock", "LlamaDecoderLayer"
    ]
    _skip_keys_device_placement = "past_key_values"  # TODO:
    is_parallelizable = False  # TODO:

    supports_gradient_checkpointing = True  # TODO:
    _supports_flash_attn_2 = True  # TODO:

    def __init__(
        self,
        config: Optional[LAMConfig] = None,
        audio_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[QWenAudioTokenizer] = None,
    ):
        if config is None and (audio_model is None or text_model is None):
            raise ValueError(
                "Either a configuration or an audio and a text model has to be provided"
            )

        if config is None:
            config = LAMConfig(audio_model.config.to_dict(),
                               text_model.config.to_dict())
        elif not isinstance(config, self.config_class):
            raise ValueError(
                f"config: {config} has to be of type {self.config_class}")

        super().__init__(config)

        if audio_model is None:
            if config.audio_config.model_type == 'whisper':
                from .modeling_whisper_encoder import WhisperEncoder
                audio_model = WhisperEncoder(config.audio_config)
            elif config.audio_config.model_type == 'whisper_openai':
                from .modeling_whisper_openai_encoder import AudioEncoder as WhisperEncoder
                audio_model = WhisperEncoder(config.audio_config)
            else:
                audio_model = AutoModel.from_config(config.audio_config)

        if text_model is None:
            text_model = AutoModelForCausalLM.from_config(
                config.text_config, trust_remote_code=True)

        self.audio_model = audio_model
        # self.text_model = text_model
        self.audio_model.config = self.config.audio_config
        # self.text_model.config = self.config.text_config

        dim_audio_emb = self.config.audio_config.d_model
        # dim_text_emb = self.config.text_config.hidden_size
        # self.m_adapter = TransformerAdaptorLayer(kernel_size=8, stride=8, dropout_p=0.1,model_dim=dim_audio_emb)
        # self.proj = nn.Linear(dim_audio_emb, dim_text_emb)  # TODO: _init_weights

        # if config.text_config.model_type == 'qwen':
        #     self.text_model_vocab_size = self.text_model.transformer.vocab_size
        # else:
        #     self.text_model_vocab_size = self.text_model.vocab_size
        # self.num_special_tokens_add = config.num_special_tokens_add
        # if self.num_special_tokens_add > 0:
        #     # self.text_model.resize_token_embeddings(new_vocab_size)
        #     # 保留了原始参数值，但无法分别指定 requires_grad
        #     self.wte_add = nn.Embedding(self.num_special_tokens_add,
        #                                 dim_text_emb)
        #     self.lm_head_add = nn.Linear(dim_text_emb,
        #                                  self.num_special_tokens_add,
        #                                  bias=False)
        #     # Initialize the weights. Copy from Qwen-Audio
        #     self.wte_add.weight.data.normal_(
        #         mean=0.0, std=self.config.text_config.initializer_range)
        #     self.lm_head_add.weight.data.normal_(
        #         mean=0.0, std=self.config.text_config.initializer_range)

        # self.to(self.text_model.dtype)  # TODO: 
        # self.to(torch.bfloat16)
        # self.wte_add.to(self.text_model.dtype)
        # self.lm_head_add.to(self.text_model.dtype)
        # self.post_init()  # TODO: _set_gradient_checkpointing
        self.tokenizer = tokenizer
        self.ctc_lo = nn.Linear(dim_audio_emb, text_model.vocab_size)  # TODO: _init_weights
        self.ctc_loss = torch.nn.CTCLoss(blank=self.tokenizer.eos_token_id,
                                         reduction='mean',
                                         zero_infinity=True)


    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).

        从 transformers.models.whisper.modeling_whisper 拷贝，修改 config 路径
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config.audio_config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.audio_config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.audio_config.mask_time_prob,
                mask_length=self.config.audio_config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.audio_config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices,
                                             device=input_features.device,
                                             dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(
                -1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.audio_config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.audio_config.mask_feature_prob,
                mask_length=self.config.audio_config.mask_feature_length,
                min_masks=self.config.audio_config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices,
                                                device=input_features.device,
                                                dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    def forward_bk(self,
                input_features: Optional[torch.FloatTensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                audio_attention_mask: Optional[torch.FloatTensor] = None,
                adapter_out_lengths:Optional[torch.LongTensor] = None,
                input_wavs: Optional[torch.FloatTensor] = None,
                raw_audio_attention_mask: Optional[torch.FloatTensor] = None,
                audio_info: Dict = None):
        '''
        不支持的输入参数：
            token_type_ids: 未采用
            position_ids: 采用 RoPE
            head_mask
            encoder_hidden_states: 用于 cross attention
            encoder_attention_mask: 用于 cross attention
        '''
        """
        Args:
            input_features (torch.Tensor): (#batch, time1, mel_size).
            input_ids (torch.Tensor): Key tensor (#batch, label_size).
            labels (torch.Tensor): Value tensor (#batch, label_size).
            audio_attention_mask: Value tensor (#batch, time1) pad is False.
            encoder_out_lengths: Value tensor (#batch).

        Returns:
        """
        if input_features is not None:
            input_features = self._mask_input_features(input_features.transpose(1, 2),
                                                       audio_attention_mask)
            audios = self.audio_model(input_features, audio_attention_mask)['last_hidden_state']

            lamda = torch.rand(1).item()
            if self.ctc_model is not None and lamda <= 0.5:
                emissions = self.ctc_model(input_wavs, raw_audio_attention_mask)['emissions']
                def get_pred(e):
                    toks = e.argmax(dim=-1).unique_consecutive()
                    hyp = [self.d2v_dict.symbols[x.item()] for x in toks[toks != 0]]
                    hyp = "".join(hyp).replace('▁', ' ').lower().strip()
                    encoding = self.tokenizer(text=hyp,
                                text_target=hyp,
                                padding=True,
                                return_tensors="pt",
                                return_attention_mask=True)
                    return encoding['input_ids'][0].to(device=input_ids.device)
                ctc_prompts = [get_pred(x) for x in emissions]
            else:
                ctc_prompts = None

            audios, padding_mask = self.m_adapter(seqs=audios, seqs_len=adapter_out_lengths)
            assert padding_mask.sum(-1).tolist() == adapter_out_lengths.tolist()

            audios = self.proj(audios)
        else:
            audios = None

        if inputs_embeds is None:
            if ctc_prompts is not None:
                input_ids_tmp = []
                ctc_prompt_lengths = []
                labels_tmp = []
                for idx in range(audio_info["audio_pos"].shape[0]):
                    ctc_prompt_len = ctc_prompts[idx].shape[0]
                    input_ids_tmp.append(torch.cat([ctc_prompts[idx], input_ids[idx]]))
                    ctc_prompt_lengths.append(ctc_prompt_len)

                    label_prefix = torch.tensor([-100]*ctc_prompt_len, 
                                                   device=labels.device, 
                                                   dtype=labels.dtype)
                    labels_tmp.append(torch.cat([label_prefix, labels[idx]]))
                input_ids = pad_sequence(input_ids_tmp,
                                batch_first=True,
                                padding_value=self.tokenizer.eos_token_id)
                
                labels = pad_sequence(labels_tmp,
                                batch_first=True,
                                padding_value=-100)
                assert input_ids.shape == labels.shape

            if self.num_special_tokens_add > 0:
                mask = input_ids >= self.text_model_vocab_size
                inputs_embeds = self.text_model.get_input_embeddings()(
                    torch.masked_fill(input_ids, mask, 0))
                emd_add = self.wte_add(
                    torch.masked_fill(input_ids - self.text_model_vocab_size,
                                      ~mask, 0))
                inputs_embeds[mask] = emd_add[mask]
            else:
                inputs_embeds = self.text_model.get_input_embeddings()(
                    input_ids)

            if audios is not None:
                for idx in range(audio_info["audio_pos"].shape[0]):
                    i, a, b = audio_info["audio_pos"][idx].tolist()
                    if ctc_prompts is not None:
                        a += ctc_prompt_lengths[idx]
                        b += ctc_prompt_lengths[idx]
                    # inputs_embeds[i][a] = bos
                    inputs_embeds[i][a + 1:b] = audios[idx][:b - a - 1]
                    # inputs_embeds[i][b] = eos

        result = self.text_model(inputs_embeds=inputs_embeds,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    labels=None,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=True,
                                    return_dict=True)
        hidden_states = result.hidden_states[-1]
        logits = result.logits
        logits_add = self.lm_head_add(hidden_states)
        lm_logits = torch.concat([logits, logits_add], dim=-1)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        return {"loss": loss,}


    def forward(self,
                input_features: Optional[torch.FloatTensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                audio_attention_mask: Optional[torch.FloatTensor] = None,
                encoder_out_lengths:Optional[torch.LongTensor] = None,
                input_wavs: Optional[torch.FloatTensor] = None,
                raw_audio_attention_mask: Optional[torch.FloatTensor] = None,
                audio_info: Dict = None):
        '''
        不支持的输入参数：
            token_type_ids: 未采用
            position_ids: 采用 RoPE
            head_mask
            encoder_hidden_states: 用于 cross attention
            encoder_attention_mask: 用于 cross attention
        '''
        """
        Args:
            input_features (torch.Tensor): (#batch, time1, mel_size).
            input_ids (torch.Tensor): Key tensor (#batch, label_size).
            labels (torch.Tensor): Value tensor (#batch, label_size).
            audio_attention_mask: Value tensor (#batch, time1) pad is False.
            encoder_out_lengths: Value tensor (#batch).

        Returns:
        """
        input_features = self._mask_input_features(input_features.transpose(1, 2),
                                                       audio_attention_mask)
        audios = self.audio_model(input_features, audio_attention_mask)['last_hidden_state']
        ys_hat = self.ctc_lo(audios)
        hlens = encoder_out_lengths

        B = input_features.shape[0]
        ys_pad = []
        ys_lens = []
        for idx in range(B):
            bos_pos = torch.where(labels[idx] != -100)[0][0].item()
            eos_pos = torch.where(labels[idx] == self.tokenizer.eos_token_id)[0][0].item()
            ys_pad.append(labels[idx][bos_pos:eos_pos])
            ys_lens.append(eos_pos - bos_pos)
        ys_pad = pad_sequence(ys_pad,batch_first=True, padding_value=-1).to(device=audios.device)
        ys_lens = torch.tensor(ys_lens, dtype=torch.int32, device=audios.device)

        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        # loss = loss / ys_hat.size(1)

        return {"loss": loss,}


    def generate(
        self,
        input_ids,
        input_features=None,
        audio_attention_mask=None,
        adapter_out_lengths:Optional[torch.LongTensor] = None,
        audio_info=None,
        input_wavs: Optional[torch.FloatTensor] = None,
        raw_audio_attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if input_features is not None:
            audios = self.audio_model(input_features.transpose(1, 2), 
                                      audio_attention_mask)["last_hidden_state"]
            audios, padding_mask = self.m_adapter(seqs=audios, seqs_len=adapter_out_lengths)
            if self.ctc_model is not None:
                emissions = self.ctc_model(input_wavs, raw_audio_attention_mask)['emissions']
                def get_pred(e):
                    toks = e.argmax(dim=-1).unique_consecutive()
                    hyp = [self.d2v_dict.symbols[x.item()] for x in toks[toks != 0]]
                    hyp = "".join(hyp).replace('▁', ' ').lower().strip()
                    encoding = self.tokenizer(text=hyp,
                                text_target=hyp,
                                padding=True,
                                return_tensors="pt",
                                return_attention_mask=True)
                    return encoding['input_ids'][0].to(device=input_ids.device)
                ctc_prompts = [get_pred(x) for x in emissions]
            else:
                ctc_prompts = None
            audios = self.proj(audios)
        else:
            audios = None

        if ctc_prompts is not None:
            input_ids_tmp = []
            ctc_prompt_lengths = []
            labels_tmp = []
            for idx in range(audio_info["audio_pos"].shape[0]):
                ctc_prompt_len = ctc_prompts[idx].shape[0]
                input_ids_tmp.append(torch.cat([ctc_prompts[idx], input_ids[idx]]).to(dtype=torch.int64))
                ctc_prompt_lengths.append(ctc_prompt_len)

            input_ids = pad_sequence(input_ids_tmp,
                            batch_first=True,
                            padding_value=self.tokenizer.eos_token_id)

        if self.num_special_tokens_add > 0:
            mask = input_ids >= self.text_model_vocab_size
            inputs_embeds = self.text_model.get_input_embeddings()(
                torch.masked_fill(input_ids, mask, 0))
            emd_add = self.wte_add(
                torch.masked_fill(input_ids - self.text_model_vocab_size,
                                  ~mask, 0))
            inputs_embeds[mask] = emd_add[mask]
        else:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if audios is not None:
            for idx in range(audio_info["audio_pos"].shape[0]):
                i, a, b = audio_info["audio_pos"][idx].tolist()
                if ctc_prompts is not None:
                    a += ctc_prompt_lengths[idx]
                    b += ctc_prompt_lengths[idx]
                # inputs_embeds[i][a] = bos
                inputs_embeds[i][a + 1:b] = audios[idx][:b - a - 1]
                # inputs_embeds[i][b] = eos
        res = self.text_model.generate(
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        assert res.size(0) == 1
        print(res)
        res = res[:,:-1] # remove eos
        if ctc_prompts is not None:
            if res[0].size(0) > ctc_prompt_lengths[0] * 2:
                res = ctc_prompts[0].unsqueeze(0)
            if res[0].size(0) * 2 < ctc_prompt_lengths[0]:
                res = ctc_prompts[0].unsqueeze(0)
        return res


    @classmethod
    def _whisper_encoder_from_pretrained(cls, model_path, config):
        from wenet.transformer.modeling_whisper_encoder import WhisperEncoder

        assert isinstance(model_path, str)
        assert isinstance(config, WhisperConfig)
        audio_model = WhisperEncoder(config)
        # old_state_dict = torch.load(
        #     os.path.join(model_path, "pytorch_model.bin"))
        # state_dict = {}
        # for para_name in old_state_dict.keys():
        #     if "model.encoder." in para_name:
        #         if "model.encoder.conv1" in para_name:
        #             continue
        #         new_name = para_name.replace("model.encoder.", "")
        #         state_dict[new_name] = old_state_dict[para_name]
                
        # audio_model.load_state_dict(state_dict, strict=False)

        return audio_model

    @classmethod
    def _whisper_encoder_from_pretrained_openai(cls, model_path, config):
        from wenet.transformer.modeling_whisper_openai_encoder import AudioEncoder as WhisperEncoder, WhisperConfig

        checkpoint = torch.load(model_path)

        dims = checkpoint["dims"]
        dims = {
            "n_mels": dims["n_mels"],
            "n_ctx": dims["n_audio_ctx"],
            "n_state": dims["n_audio_state"],
            "n_head": dims["n_audio_head"],
            "n_layer": dims["n_audio_layer"]
        }
        audio_pretrained_config = WhisperConfig(**dims, **config)

        old_state_dict = checkpoint["model_state_dict"]
        state_dict = {}
        for para_name in old_state_dict.keys():
            if "encoder." in para_name:
                new_name = para_name.replace("encoder.", "")
                state_dict[new_name] = old_state_dict[para_name]

        audio_model = WhisperEncoder(audio_pretrained_config)
        audio_model.load_state_dict(state_dict)

        return audio_model, audio_pretrained_config

    @classmethod
    def from_audio_text_pretrained(
        cls,
        audio_model,
        text_model,
        audio_config: Optional[Dict] = {},
        text_config: Optional[Dict] = {},
        num_special_tokens_add: Optional[int] = 0,
        tokenizer: QWenAudioTokenizer = None,
    ) -> PreTrainedModel:
        """
        参考 from_vision_text_pretrained
        """
        ctc_model = None
        if audio_config.get("model_type") == "whisper_openai":
            audio_model, audio_pretrained_config = cls._whisper_encoder_from_pretrained_openai(
                audio_model, audio_config)
            
        else:
            # audio_config['d2v_path'] = None
            # d = None
            # if audio_config['d2v_path'] is not None:
            #     cfg = Wav2Vec2CtcConfig(**audio_config['model'])
            #     state = checkpoint_utils.load_checkpoint_to_cpu(audio_config['d2v_path'])

            #     d = state['task_state']['target_dictionary']
            #     ctc_model = Wav2VecCtc.build_model(cfg, d)
            #     ctc_model.load_state_dict(state['model'], strict=False)

            #     audio_config.pop('d2v_path')
            #     audio_config.pop('model')

            audio_pretrained_config = AutoConfig.from_pretrained(
                audio_model, **audio_config)

            if audio_pretrained_config.model_type == "whisper":
                audio_model = cls._whisper_encoder_from_pretrained(
                    audio_model, audio_pretrained_config)
            else:
                audio_model = AutoModel.from_pretrained(
                    audio_model, **audio_config)

        text_pretrained_config = AutoConfig.from_pretrained(
            text_model, **text_config, trust_remote_code=True)
        text_model = AutoModelForCausalLM.from_pretrained(
            text_model, trust_remote_code=True, **text_config)

        lam_config = LAMConfig(audio_pretrained_config.to_dict(),
                               text_pretrained_config.to_dict(),
                               num_special_tokens_add=num_special_tokens_add)
        model = cls(config=lam_config,
                    audio_model=audio_model,
                    text_model=text_model,
                    tokenizer=tokenizer)
        model.generation_config = text_model.generation_config
        return model
