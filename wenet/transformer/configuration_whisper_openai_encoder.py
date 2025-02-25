from transformers import PretrainedConfig


class WhisperConfig(PretrainedConfig):
    """
    n_mels (`int`, *optional*, defaults to 80):
        Number of mel features used per input features. Should correspond to the value used in the
        `WhisperProcessor` class.
    n_layer (`int`, *optional*, defaults to 4):
        Number of encoder layers.
    n_head (`int`, *optional*, defaults to 6):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_state (`int`, *optional*, defaults to 384):
        Dimensionality of the layers.
    n_ctx (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    apply_spec_augment (`bool`, *optional*, defaults to `False`):
        Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
        [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
        Recognition](https://arxiv.org/abs/1904.08779).
    mask_time_prob (`float`, *optional*, defaults to 0.05):
        Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
        procecure generates `mask_time_prob*len(time_axis)/mask_time_length` independent masks over the axis. If
        reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
        masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
        actual percentage of masked vectors. This is only relevant if `apply_spec_augment == True`.
    mask_time_length (`int`, *optional*, defaults to 10):
        Length of vector span along the time axis.
    mask_time_min_masks (`int`, *optional*, defaults to 2),:
        The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
        irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
        mask_time_min_masks''
    mask_feature_prob (`float`, *optional*, defaults to 0.0):
        Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
        masking procecure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
        the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
        span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
        may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
        True`.
    mask_feature_length (`int`, *optional*, defaults to 10):
        Length of vector span along the feature axis.
    mask_feature_min_masks (`int`, *optional*, defaults to 0),:
        The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
        step, irrespectively of `mask_feature_prob`. Only relevant if
        `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
    """

    model_type = "whisper_openai"  # 用于保存模型时自动保存对应的代码
    _auto_class = "AutoConfig"  # 用于保存模型时自动保存对应的代码

    def __init__(
        self,
        n_mels: int = 80,
        n_ctx: int = 1500,
        n_state: int = 384,
        n_head: int = 6,
        n_layer: int = 4,
        apply_spec_augment: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        **kwargs,
    ):
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks
        self.d_model = n_state
        super().__init__(**kwargs)
