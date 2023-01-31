import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from utils.tools import get_mask_from_lengths, pad, word_level_pooling

from .blocks import (
    ConvNorm,
    RelativeFFTBlock,
    WordToPhonemeAttention,
)
from text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class LinguisticEncoder(nn.Module):     # 先通过声素编码器 -> 词编码器 -> 带有持续时间预测的LR -> word-phoneme-attention
    """ Linguistic Encoder """

    def __init__(self, preprocess_config, model_config, train_config):
        super(LinguisticEncoder, self).__init__()

        n_position = model_config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = model_config["transformer"]["encoder_hidden"]
        n_layers = model_config["transformer"]["encoder_layer"]
        n_head = model_config["transformer"]["encoder_head"]
        d_k = d_v = (
            model_config["transformer"]["encoder_hidden"]
            // model_config["transformer"]["encoder_head"]
        )
        d_model = model_config["transformer"]["encoder_hidden"]
        d_inner = model_config["transformer"]["conv_filter_size"]
        kernel_size = model_config["transformer"]["conv_kernel_size"]
        dropout = model_config["transformer"]["encoder_dropout"]
        window_size = model_config["transformer"]["encoder_window_size"]
        self.helper_type = train_config["aligner"]["helper_type"]

        self.max_seq_len = model_config["max_seq_len"]
        self.d_model = d_model
        self.n_head = n_head

        self.src_emb = nn.Embedding(                    # 音素位置编码
            n_src_vocab, d_word_vec, padding_idx=0
        )
        self.abs_position_enc = nn.Parameter(           # 音素位置编码
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.kv_position_enc = nn.Parameter(            # 音素kv位置编码
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=True,
        )
        self.q_position_enc = nn.Parameter(             # 音素q位置编码
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=True,
        )
                                                        # 位置编码后经过声素编码器     声素编码器将音素编码序列转为音素隐藏状态
        self.phoneme_encoder = RelativeFFTBlock(        # 音素编码器   FFT块 + 多头注意机制
            hidden_channels=d_model,
            filter_channels=d_inner,
            n_heads=n_head,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=dropout,
            window_size=window_size,
        )                                               # 声素编码器输出后经过词级池化（WP）后进入词编码器  进入带有预测时间的长度调节器
        self.word_encoder = RelativeFFTBlock(           # 词编码器    FFT块 + 多头注意机制
            hidden_channels=d_model,
            filter_channels=d_inner,
            n_heads=n_head,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=dropout,
            window_size=window_size,
        )
        self.length_regulator = LengthRegulator()       # 长度调节器
        self.duration_predictor = VariancePredictor(model_config)   # 方差预测器   -- 持续时间预测器    FastSpeech2中的方差预测器
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.w2p_attn = WordToPhonemeAttention(         # 词-音素注意机制
            n_head, d_model, d_k, d_v  # , dropout=dropout
        )
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(  # TextGrid  事先用MFA训练好的音素持续时间预测
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):        # (1,x,256), (1,36), (1,36), 1.0                              # 看不太懂target
        prediction = self.pitch_predictor(x, mask)         # 经过Variance Prediction输出后的一个预测时间
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))          # 对pitch进行字符嵌入
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)         # 经过Variance Prediction输出后的一个预测时间
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def get_mapping_mask(self, q, kv, dur_w, wb, src_w_len):
        """
        For applying a word-to-phoneme mapping mask to the attention weight to force each query (Q) 
        to only attend to the phonemes belongs to the word corresponding to this query.
        """
        batch_size, q_len, kv_len, device = q.shape[0], q.shape[1], kv.shape[1], kv.device
        mask = torch.ones(batch_size, q_len, kv_len, device=device)
        for b, (w, p, l) in enumerate(zip(dur_w, wb, src_w_len)):
            w, p = [0]+[d.item() for d in torch.cumsum(w[:l], dim=0)], [0] + \
                [d.item() for d in torch.cumsum(p[:l], dim=0)]
            # assert len(w) == len(p)
            for i in range(1, len(w)):
                mask[b, w[i-1]:w[i], p[i-1]:p[i]
                     ] = torch.zeros(w[i]-w[i-1], p[i]-p[i-1], device=device)
        return mask == 0.

    def add_position_enc(self, src_seq, position_enc=None, coef=None):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            pos_enc = get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
            if coef is not None:
                pos_enc = coef.unsqueeze(-1) * pos_enc
            enc_output = src_seq + pos_enc
        else:
            position_enc = self.abs_position_enc if position_enc is None else position_enc
            pos_enc = position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            if coef is not None:
                pos_enc = coef.unsqueeze(-1) * pos_enc
            enc_output = src_seq + pos_enc
        return enc_output

    def get_rel_coef(self, dur, dur_len, mask):
        """
        For adding a well-designed positional encoding to the inputs of word-to-phoneme attention module.
        """
        idx, L, device = [], [], dur.device
        for d, dl in zip(dur, dur_len):
            idx_b, d = [], d[:dl].long()
            m = torch.repeat_interleave(d, torch.tensor(
                list(d), device=device), dim=0)  # [tgt_len]
            L.append(m)
            for d_i in d:
                idx_b += list(range(d_i))
            idx.append(torch.tensor(idx_b).to(device))
            # assert L[-1].shape == idx[-1].shape
        return torch.div(pad(idx).to(device), pad(L).masked_fill(mask == 0., 1.).to(device))

    def forward(
        self,
        src_p_seq,  # (8,x,192)
        src_p_len,
        word_boundary,
        src_p_mask, # (8,x)
        src_w_len,
        src_w_mask,
        mel_mask=None,
        max_len=None,
        attn_prior=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        duration_control=1.0,
    ):

        # Phoneme Encoding                                      # 声素编码器 -> 经过WP词级池化到传入词编码器
        src_p_seq = self.src_emb(src_p_seq)         # (1,40) -> (1,40,256)
        # t=src_p_mask.unsqueeze(1)   (8,1,4)
        enc_p_out = self.phoneme_encoder(src_p_seq.transpose(
            1, 2), src_p_mask.unsqueeze(1)).transpose(1, 2)         # (8,x,192)

        # Pitch、Energy Predictor
        if self.pitch_feature_level == "phoneme_level":     # 此时的x为编码器的输出(1,185,256)
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                enc_p_out, pitch_target, src_p_mask, p_control  # (1,185,256), (1,20), (1,20)   pitch_target和src_mask维度相同
            )
            enc_p_out = enc_p_out + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                enc_p_out, energy_target, src_p_mask, p_control
            )
            enc_p_out = enc_p_out + energy_embedding

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                enc_p_out, pitch_target, mel_mask, p_control
            )
            enc_p_out = enc_p_out + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                enc_p_out, energy_target, mel_mask, p_control
            )
            enc_p_out = enc_p_out + energy_embedding

        # Word-level Pooing
        src_w_seq = word_level_pooling(                         # WP词级池化 -> 词编码器
            enc_p_out, src_p_len, word_boundary, src_w_len, reduce="mean")

        # Word Encoding
        enc_w_out = self.word_encoder(src_w_seq.transpose(      # 词编码器 -> 持续时间预测器
            1, 2), src_w_mask.unsqueeze(1)).transpose(1, 2)

        # Phoneme-level Duration Prediction
        log_duration_p_prediction = self.duration_predictor(    # 持续时间预测器 -> WP词级池化
            enc_p_out, src_p_mask)      # (1,27)
                                        # (1.14)
        # Word-level Pooling (in log scale)
        log_duration_w_prediction = word_level_pooling(
            log_duration_p_prediction.exp().unsqueeze(-1), src_p_len, word_boundary, src_w_len, reduce="sum").log().squeeze(-1)

        x = enc_w_out
        if duration_target is not None:                         # WP词级池化 -> LR长度调节器
            # Word-level Pooing
            duration_w_rounded = word_level_pooling(
                duration_target.unsqueeze(-1), src_p_len, word_boundary, src_w_len, reduce="sum").squeeze(-1)
            # Word-level Length Regulate
            x, mel_len = self.length_regulator(x, duration_w_rounded, max_len)
        else:
            # Word-level Duration
            duration_w_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_w_prediction) - 1) * duration_control),
                min=0,
            ).long()
            # Word-level Length Regulate
            x, mel_len = self.length_regulator(x, duration_w_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        # Word-to-Phoneme Attention
        # [batch, mel_len, seq_len]
        src_mask_ = src_p_mask.unsqueeze(1).expand(-1, mel_mask.shape[1], -1)
        # [batch, mel_len, seq_len]
        mel_mask_ = mel_mask.unsqueeze(-1).expand(-1, -1, src_p_mask.shape[1])
        # [batch, mel_len, seq_len]
        mapping_mask = self.get_mapping_mask(
            x, enc_p_out, duration_w_rounded, word_boundary, src_w_len) 

        q = self.add_position_enc(x, position_enc=self.q_position_enc, coef=self.get_rel_coef(
            duration_w_rounded, src_w_len, mel_mask))
        k = self.add_position_enc(
            enc_p_out, position_enc=self.kv_position_enc, coef=self.get_rel_coef(word_boundary, src_p_len, src_p_mask))
        v = self.add_position_enc(
            enc_p_out, position_enc=self.kv_position_enc, coef=self.get_rel_coef(word_boundary, src_p_len, src_p_mask))
        # q = self.add_position_enc(x)                          k、v为音素编码器的输出   q大概率为LR的输出
        # k = self.add_position_enc(enc_p_out)
        # v = self.add_position_enc(enc_p_out)
        x, attns, attn_logprob = self.w2p_attn(                # word-to-phoneme attention  q、k、v为输入
            q=q,
            k=k,
            v=v,
            key_mask=src_mask_,
            query_mask=mel_mask_,
            mapping_mask=mapping_mask,
            indivisual_attn=True,
            attn_prior=attn_prior if self.helper_type == "ctc" else None,
        )

        # # Pitch、Energy Predictor
        # if self.pitch_feature_level == "phoneme_level":     # 此时的x为编码器的输出(1,185,256)
        #     pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #         enc_w_out, pitch_target, src_p_mask, p_control  # (1,185,256), (1,20), (1,20)   pitch_target和src_mask维度相同
        #     )
        #     x = x + pitch_embedding
        # if self.energy_feature_level == "phoneme_level":
        #     energy_prediction, energy_embedding = self.get_energy_embedding(
        #         enc_w_out, energy_target, src_p_mask, p_control
        #     )
        #     x = x + energy_embedding
        #
        # if self.pitch_feature_level == "frame_level":
        #     pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #         enc_w_out, pitch_target, mel_mask, p_control
        #     )
        #     x = x + pitch_embedding
        # if self.energy_feature_level == "frame_level":
        #     energy_prediction, energy_embedding = self.get_energy_embedding(
        #         enc_w_out, energy_target, mel_mask, p_control
        #     )
        #     x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_w_prediction,
            duration_w_rounded,
            mel_len,
            mel_mask,
            attns,
            attn_logprob,
        )


class LengthRegulator(nn.Module):       # LR 长度调节器
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):     # LR传入参数： x字符  持续时间  最大长度
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):             # 同FastSpeech2中的预测器
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvNorm(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            channel_last=True,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        ConvNorm(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=1,
                            dilation=1,
                            channel_last=True,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)   # (1,32,256)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out * mask

        return out
