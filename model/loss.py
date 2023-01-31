import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
from utils.tools import ssim
from utils.pitch_tools import cwt2f0_norm
from text import sil_phonemes_ids


def get_lsgan_losses_fn():

    def jcu_loss_fn(logit_cond, logit_uncond, label_fn, mask=None):
        cond_loss = F.mse_loss(logit_cond, label_fn(logit_cond), reduction="none" if mask is not None else "mean")
        cond_loss = (cond_loss * mask).sum() / mask.sum() if mask is not None else cond_loss
        uncond_loss = F.mse_loss(logit_uncond, label_fn(logit_uncond), reduction="none" if mask is not None else "mean")
        uncond_loss = (uncond_loss * mask).sum() / mask.sum() if mask is not None else uncond_loss
        return 0.5 * (cond_loss + uncond_loss)

    def d_loss_fn(r_logit_cond, r_logit_uncond, f_logit_cond, f_logit_uncond, mask=None):
        r_loss = jcu_loss_fn(r_logit_cond, r_logit_uncond, torch.ones_like, mask)
        f_loss = jcu_loss_fn(f_logit_cond, f_logit_uncond, torch.zeros_like, mask)
        return r_loss, f_loss

    def g_loss_fn(f_logit_cond, f_logit_uncond, mask=None):
        f_loss = jcu_loss_fn(f_logit_cond, f_logit_uncond, torch.ones_like, mask)
        return f_loss

    return d_loss_fn, g_loss_fn

def get_adversarial_losses_fn(mode):
    if mode == 'lsgan':
        return get_lsgan_losses_fn()
    else:
        raise NotImplementedError


class MixGANTTSLoss(nn.Module):
    """ MixGAN-TTS Loss """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(MixGANTTSLoss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()        # 均方误差损失
        self.model = args.model
        self.loss_config = train_config["loss"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.helper_type = train_config["aligner"]["helper_type"]
        if self.helper_type == "dga":
            self.guided_attn_loss = GuidedAttentionLoss(
                sigma=train_config["aligner"]["guided_sigma"],
                alpha=train_config["aligner"]["guided_lambda"],
            )
            self.guided_attn_weight = train_config["aligner"]["guided_weight"]
        elif self.helper_type == "ctc":
            self.sum_loss = ForwardSumLoss()
            self.ctc_step = train_config["step"]["ctc_step"]
            self.ctc_weight_start = train_config["aligner"]["ctc_weight_start"]
            self.ctc_weight_end = train_config["aligner"]["ctc_weight_end"]

        self.n_layers = model_config["discriminator"]["n_layer"] + \
                        model_config["discriminator"]["n_cond_layer"]
        self.lambda_d = train_config["loss"]["lambda_d"]
        self.lambda_p = train_config["loss"]["lambda_p"]
        self.lambda_e = train_config["loss"]["lambda_e"]
        self.lambda_fm = train_config["loss"]["lambda_fm" if self.model != "shallow" else "lambda_fm_shallow"]
        self.sil_ph_ids = sil_phonemes_ids()
        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn(train_config["loss"]["adv_loss_mode"])

    def mle_loss(self, z, logdet, mask):  # (8,80,x), tensor(8), (8,1,x)
        """
        z, logdet: [batch_size, dim, max_time]
        mask -- [batch_size, 1, max_time]
        """
        logs = torch.zeros_like(z * mask)
        l = torch.sum(logs) + 0.5 * \
            torch.sum(torch.exp(-2 * logs) * (z ** 2))
        l = l - torch.sum(logdet)
        l = l / \
            torch.sum(torch.ones_like(z * mask))
        l = l + 0.5 * math.log(2 * math.pi)
        return l

    def forward(self, model, inputs, predictions, step, coarse_mels=None, Ds=None):
        (
            texts,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            mel_targets,    # 11
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[3:]
        (   # 16
            mel_predictions,
            _,
            _,
            _,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_roundeds,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            alignments,
            alignment_logprobs,
            src_w_masks,  # true
            postnet_output,
        ) = predictions
        # src_masks = ~src_masks # (8,x) 取反  False -> True  目前True居多
        mel_masks = ~mel_masks      # (8,x) 取反  False -> True  目前True居多
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]   # (8,x,80)
        log_duration_targets = torch.log(duration_roundeds.float() + 1)
        self.mel_masks = mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks

        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        log_duration_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_w_masks)        #(1,21), (1,39)
        log_duration_targets = log_duration_targets.masked_select(src_w_masks)

        # Acoustic reconstruction loss
        if self.model == "aux":
            # Get post_loss
            #postnet_output = postnet_output.masked_select(mel_masks.unsqueeze(-1))
            postnet_loss = self.mae_loss(postnet_output, mel_targets)

            mel_loss = torch.zeros(1).to(mel_targets.device)
            for _mel_predictions in mel_predictions:
                _mel_predictions = model.module.diffusion.denorm_spec(_mel_predictions) # since we normalize mel in diffuse_trace
                mel_loss += self.get_mel_loss(_mel_predictions, mel_targets)
        elif self.model == "shallow":
            # Get post_loss
            # postnet_output = postnet_output.masked_select(mel_masks.unsqueeze(-1))
            postnet_loss = self.mae_loss(postnet_output, mel_targets)

            coarse_mels = coarse_mels[:, : mel_masks.shape[1], :]
            mel_predictions = model.module.diffusion.denorm_spec(mel_predictions) # since we use normalized mel
            mel_loss = self.get_mel_loss(mel_predictions, coarse_mels.detach())
        elif self.model == "naive":
            assert coarse_mels is None
            mel_predictions = model.module.diffusion.denorm_spec(mel_predictions) # since we use normalized mel
            mel_loss = self.get_mel_loss(mel_predictions, mel_targets)
            postnet_loss = torch.zeros(1).to(mel_targets.device)

        duration_loss, pitch_loss, energy_loss = self.get_init_losses(mel_targets.device)
        helper_loss = attn_loss = ctc_loss = torch.zeros(1).to(mel_targets.device)
        if self.model != "shallow":
            duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
            pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
            energy_loss = self.mse_loss(energy_predictions, energy_targets)

            if self.helper_type == "dga":
                for alignment in alignments[1]:  # DGA should be applied on attention without mapping mask
                    attn_loss += self.guided_attn_loss(alignment, src_lens, mel_lens)
                # attn_loss = self.guided_attn_loss(alignments[1][0], src_lens, mel_lens)
                helper_loss = self.guided_attn_weight * attn_loss
            elif self.helper_type == "ctc":
                for alignment_logprob in alignment_logprobs:
                    ctc_loss += self.sum_loss(alignment_logprob, src_lens, mel_lens)
                ctc_loss = ctc_loss.mean()
                helper_loss = (self.ctc_weight_start if step <= self.ctc_step else self.ctc_weight_end) * ctc_loss
        recon_loss = mel_loss + postnet_loss + self.lambda_d * duration_loss + self.lambda_p * pitch_loss + \
                     self.lambda_e * energy_loss + helper_loss
        # recon_loss = mel_loss + postnet_loss + duration_loss + pitch_loss + energy_loss + helper_loss

        # Feature matching loss
        fm_loss = torch.zeros(1).to(mel_targets.device)
        if Ds is not None:
            fm_loss = self.lambda_fm * self.get_fm_loss(*Ds)
            # self.lambda_fm = recon_loss.item() / fm_loss.item() # dynamic scaling following (Yang et al., 2021)

        return (
            fm_loss,
            recon_loss,
            mel_loss,
            postnet_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            helper_loss,
        )

    def get_init_losses(self, device):
        duration_loss = torch.zeros(1).to(device)
        pitch_loss = torch.zeros(1).to(device)
        energy_loss = torch.zeros(1).to(device)
        return duration_loss, pitch_loss, energy_loss

    def get_fm_loss(self, D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond):
        loss_fm = 0
        feat_weights = 4.0 / (self.n_layers + 1)
        for j in range(len(D_fake_cond) - 1):
            loss_fm += feat_weights * \
                0.5 * (F.l1_loss(D_real_cond[j].detach(), D_fake_cond[j]) + F.l1_loss(D_real_uncond[j].detach(), D_fake_uncond[j]))
        return loss_fm

    def get_mel_loss(self, mel_predictions, mel_targets):
        mel_targets.requires_grad = False
        mel_predictions = mel_predictions.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_targets = mel_targets.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_loss = self.l1_loss(mel_predictions, mel_targets)
        return mel_loss

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    See https://github.com/espnet/espnet/blob/e962a3c609ad535cd7fb9649f9f9e9e0a2a27291/espnet/nets/pytorch_backend/e2e_tts_tacotron2.py#L25
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = self.make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = self.make_non_pad_mask(olens)  # (B, T_out)
        # (B, T_out, T_in)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

    def make_non_pad_mask(self, lengths, xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, xs, length_dim)

    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask

class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss
