import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import LinearNorm, ConvNorm, DiffusionEmbedding, Mish
from transformer import PostNet, Decoder
from .linguistic_encoder import LinguisticEncoder
from .diffusion import GaussianDiffusion
from utils.tools import get_mask_from_lengths
from .loss import get_adversarial_losses_fn


class DiffGANTTS(nn.Module):                                # 扩散解码器在同一目录的modules.py最下方
    """ DiffGAN-TTS """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffGANTTS, self).__init__()
        self.model = args.model
        self.model_config = model_config

        self.linguistic_encoder = LinguisticEncoder(preprocess_config, model_config, train_config)     # 语言编码器  -> 变分发生器
        if self.model in ["aux", "shallow"]:                                                    # 在DiffGAN-TTS直接运用音高、能量轮廓
            self.decoder = Decoder(model_config)
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
            self.postnet = PostNet()
        self.diffusion = GaussianDiffusion(args, preprocess_config, model_config, train_config) # 高斯分布建模降噪分布

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )

    def forward(    # 19
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        word_boundaries,
        src_w_lens,
        max_src_w_len,
        speak_embeds=None,
        attn_priors=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        spker_embeds=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)    # 获得字符掩码
        src_w_masks = get_mask_from_lengths(src_w_lens, max_src_w_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)            # 获得频谱掩码
            if mel_lens is not None
            else None
        )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            alignments,
            alignment_logprobs,
        ) = self.linguistic_encoder(
            texts,
            src_lens,
            word_boundaries,
            src_masks,
            src_w_lens,
            src_w_masks,
            mel_masks,
            max_mel_len,
            attn_priors,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            d_control,
        )

        speaker_emb = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_emb = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_emb = self.speaker_emb(spker_embeds) # [B, H]

        if self.model == "naive":
            mel_masks = ~mel_masks
            coarse_mels = postnet_outputs = None
            (
                output, # x_0_pred
                x_ts,
                x_t_prevs,
                x_t_prev_preds,
                diffusion_step,
            ) = self.diffusion(  # 如果是naive 就用高斯分布建模噪声分布    传入mel_mask多为Flase
                mels,
                output,
                speaker_emb,
                mel_masks,
            )
        elif self.model in ["aux", "shallow"]:
            mel_masks = ~mel_masks
            x_ts = x_t_prevs = x_t_prev_preds = diffusion_step = None
            cond = output.clone()  # 复制一份FastSpeech2的Encoder和Variance adaptor 活性浅层扩散模型 复制加冻结
            coarse_mels = self.decoder(output, mel_masks)  # FastSpeech2 Decoder合成粗糙的mel谱图 作为扩散模型是条件输入 (8,x,256)
            coarse_mels = self.mel_linear(coarse_mels)  # Decoder后加linear
            postnet_outputs= self.postnet(coarse_mels) + coarse_mels
            coarse_mels = postnet_outputs
            if self.model == "aux":
                output = self.diffusion.diffuse_trace(coarse_mels, mel_masks)  # 传入mel谱 和 频谱掩码 False多
            elif self.model == "shallow":
                (
                    output,  # x_0_pred
                    x_ts,
                    x_t_prevs,
                    x_t_prev_preds,
                    diffusion_step,
                ) = self.diffusion(  # 看不懂
                    mels,
                    self._detach(cond),
                    self._detach(speaker_emb),
                    self._detach(mel_masks),
                    self._detach(coarse_mels),
                )
        else:
            raise NotImplementedError

        return [    # 16
            output,
            (x_ts, x_t_prevs, x_t_prev_preds),      # 扩散模型输出
            self._detach(speaker_emb),
            diffusion_step,
            p_predictions, # cannot detach each value in dict but no problem since loss will not use it
            self._detach(e_predictions),
            log_d_predictions, # cannot detach each value in dict but no problem since loss will not use it
            self._detach(d_rounded),
            self._detach(src_masks),
            self._detach(mel_masks),
            self._detach(src_lens),
            self._detach(mel_lens),
            alignments,
            alignment_logprobs,
            src_w_masks,
            postnet_outputs,
        ], p_targets, self._detach(coarse_mels)

    def _detach(self, p):
        return p.detach() if p is not None and self.model == "shallow" else p


class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, preprocess_config, model_config, train_config):
        super(JCUDiscriminator, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        n_layer = model_config["discriminator"]["n_layer"]
        n_uncond_layer = model_config["discriminator"]["n_uncond_layer"]
        n_cond_layer = model_config["discriminator"]["n_cond_layer"]
        n_channels = model_config["discriminator"]["n_channels"]
        kernel_sizes = model_config["discriminator"]["kernel_sizes"]
        strides = model_config["discriminator"]["strides"]
        self.multi_speaker = model_config["multi_speaker"]

        self.input_projection = LinearNorm(2 * n_mel_channels, 2 * n_mel_channels)  # 线性加激活函数
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)            # diffusion step embedding
        self.mlp = nn.Sequential(                                                   # 线性 Mish 线性  diffusion step embedding
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),                                                                 # return x * torch.tanh(F.softplus(x))
            LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        )
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(
                LinearNorm(residual_channels, n_channels[n_layer-1]),
            )
        self.conv_block = nn.ModuleList(    # 3层conv1D  LeakyReLU
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer) # 3层conv1D  LeakyReLU
            ]
        )
        self.uncond_conv_block = nn.ModuleList( # 两层conv1D 无条件卷积输出
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)   #（3，5）
            ]
        )
        self.cond_conv_block = nn.ModuleList(   # 两层conv1D 条件卷积输出
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)     #（3，5）
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, s, t):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x = self.input_projection(                  # 线性加激活函数
            torch.cat([x_t_prevs, x_ts], dim=-1)
        ).transpose(1, 2)
        diffusion_step = self.mlp(self.diffusion_embedding(t)).unsqueeze(-1)    # diffusion step embedding
        if self.multi_speaker:
            speaker_emb = self.spk_mlp(s).unsqueeze(-1)

        cond_feats = []                             # 条件特征
        uncond_feats = []                           # 非条件特征
        for layer in self.conv_block:               # layer为单层卷积层   用leaky_relu线性激活
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)                    # 添加条件和非条件特征  -- 经过卷积层后的特征
            uncond_feats.append(x)

        x_cond = (x + diffusion_step + speaker_emb) \
            if self.multi_speaker else (x + diffusion_step) # 条件输出-- 经过Conv1D Block（三层卷积层 + leaky_relu）卷积后的特征加上扩散步嵌入
        x_uncond = x                                        # 无条件输出-经过Conv1D Block的输出特征

        for layer in self.cond_conv_block:           # 条件块  2个1D卷积 + leakyReLU
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:    # 非条件块 2个1D卷积 + leakyReLU
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats
