from classification import SequenceClassificationLayer, TokenClassificationLayer
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
from mixer import Mixer


class TPMixerSeqCls(nn.Module):
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        seq_cls_cfg: DictConfig,
        topic_cfg: DictConfig,
        **kwargs
    ):
        super(TPMixerSeqCls, self).__init__(**kwargs)
        self.TP_mixer = TPMixer(mixer_cfg)
        self.seq_cls = SequenceClassificationLayer(**seq_cls_cfg)
        self.pipeline1 = nn.Sequential(
            nn.Linear(topic_cfg.topic_num, bottleneck_cfg.hidden_dim),
            nn.GELU(),
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(bottleneck_cfg.feature_size, bottleneck_cfg.hidden_dim),
            nn.GELU(),
        )

    def forward(self, inputs1: torch.Tensor, inputs2:torch.Tensor) -> torch.Tensor:
        inputs1 = inputs1.to(torch.float32)
        inputs1 = self.pipeline1(inputs1)
        inputs2 = self.pipeline2(inputs2)
        input = inputs1 + inputs2

        reprs = self.TP_mixer(input)
        seq_logits = self.seq_cls(reprs)
        return seq_logits


class TPMixerTokenCls(nn.Module):
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        token_cls_cfg: DictConfig,
        topic_cfg: DictConfig,
        **kwargs
    ):
        super(TPMixerTokenCls, self).__init__(**kwargs)
        self.TP_mixer = TPMixer(mixer_cfg)
        self.token_cls = TokenClassificationLayer(**token_cls_cfg)
        self.TP_mixer = TPMixer(mixer_cfg)
        self.pipeline1 = nn.Sequential(
            nn.Linear(topic_cfg.topic_num, bottleneck_cfg.hidden_dim),
            nn.GELU(),
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(bottleneck_cfg.feature_size, bottleneck_cfg.hidden_dim),
            nn.GELU(),
        )

    def forward(self, inputs1: torch.Tensor, inputs2:torch.Tensor) -> torch.Tensor:
        inputs1 = self.pipeline1(inputs1)
        inputs2 = self.pipeline2(inputs2)
        input = inputs1 + inputs2
        reprs = self.TP_mixer(input)
        print(reprs.shape)
        token_logits = self.token_cls(reprs)
        return token_logits


class TPMixer(nn.Module):
    def __init__(
        self,
        mixer_cfg: DictConfig,
        **kwargs
    ):
        super(TPMixer, self).__init__(**kwargs)
        self.mixer = Mixer(**mixer_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reprs = self.mixer(inputs)
        return reprs