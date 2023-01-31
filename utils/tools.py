import os
import json
import yaml
from math import exp

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "rb"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "rb"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "rb"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config

def to_device(data, device):
    if len(data) == 17:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            word_boundaries,
            src_w_lens,
            max_src_w_len,
            spker_embeds,
            attn_priors,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        word_boundaries = torch.from_numpy(word_boundaries).long().to(device)
        src_w_lens = torch.from_numpy(src_w_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)
        attn_priors = torch.from_numpy(attn_priors).float().to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            word_boundaries,
            src_w_lens,
            max_src_w_len,
            spker_embeds,
            attn_priors,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ]

    if len(data) == 10:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            word_boundaries,
            src_w_lens,
            max_src_w_len,
            spker_embeds
        ) = data
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        word_boundaries = torch.from_numpy(word_boundaries).long().to(device)
        src_w_lens = torch.from_numpy(src_w_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, word_boundaries, src_w_lens, max_src_w_len, spker_embeds)


def log(
    logger, step=None, losses=None, lr=None, figs=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/D_loss", losses[1], step)
        logger.add_scalar("Loss/G_loss", losses[2], step)
        logger.add_scalar("Loss/recon_loss", losses[3], step)
        logger.add_scalar("Loss/fm_loss", losses[4], step)
        logger.add_scalar("Loss/adv_loss", losses[5], step)
        logger.add_scalar("Loss/mel_loss", losses[6], step)
        logger.add_scalar("Loss/postnet_loss", losses[7], step)
        logger.add_scalar("Loss/pitch_loss", losses[8], step)
        logger.add_scalar("Loss/energy_loss", losses[9], step)
        logger.add_scalar("Loss/duration_loss", losses[10], step)
        logger.add_scalar("Loss/helper_loss", losses[11], step)

    if lr is not None:
        logger.add_scalar("Training/learning_rate", lr, step)

    if figs is not None:
        logger.add_figure(tag, figs, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step,
            sample_rate=sampling_rate,
        )

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return ~mask

def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(args, targets, predictions, coarse_mels, vocoder, model_config, preprocess_config, diffusion):
    timesteps = model_config["denoiser"]["timesteps" if args.model == "naive" else "shallow_timesteps"]
    basename = targets[0][0]
    src_len = predictions[10][0].item()
    mel_len = predictions[11][0].item()
    alignment = predictions[12][0][:, 0, :mel_len,
                               :src_len].float().detach().transpose(-2, -1)
    mel_target = targets[11][0, :mel_len].float().detach().transpose(0, 1)
    duration = targets[16][0, :src_len].int().detach().cpu().numpy()
    figs = {}

    if args.model == "aux":
        # denormalizing x_0 is needed due to diffuse_trace
        mel_prediction = diffusion.denorm_spec(predictions[0])[0, :mel_len].float().detach().transpose(0, 1)
        mels = [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
        ]
        titles = ["Sampled Spectrogram", "GT"]
    else:
        mels = [mel_pred[0, :mel_len].float().detach().transpose(0, 1) for mel_pred in diffusion.sampling()]
        mel_prediction = mels[-1]
        if args.model == "shallow":
            coarse_mel = coarse_mels[0, :mel_len].float().detach().transpose(0, 1)
            mels.append(coarse_mel)
        mels.append(mel_target)
        titles = [f"T={t}" if t!=0 else f"T={t}" for t in range(0, timesteps+1)[::-1]] \
            + (["Coarse Spectrogram"] if args.model == "shallow" else []) + ["GT"]
        diffusion.aux_mel = None

    figs = plot_mel(mels, titles)

    attn_fig = plot_multi_attn(
        [
            alignment.cpu().numpy(),
        ],
        # ["Word-to-Phoneme Attention Alignment"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return figs, attn_fig, wav_reconstruction, wav_prediction, basename


def synth_samples(args, targets, predictions, vocoder, model_config, preprocess_config, path, diffusion):

    multi_speaker = model_config["multi_speaker"]
    teacher_forced_tag = "_teacher_forced" if args.teacher_forced else ""
    basenames = targets[0]
    if args.model == "aux":
        # denormalizing x_0 is needed due to diffuse_trace
        predictions[0] = diffusion.denorm_spec(predictions[0][0])
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[10][i].item()
        mel_len = predictions[11][i].item()
        mel_prediction = predictions[0][i, :mel_len].detach().transpose(0, 1)   # (80,111)
        duration = predictions[7][i, :src_len].detach().cpu().numpy()

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}{}.png".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.png".format(basename, teacher_forced_tag))
        fig = plot_mel(
            [
                mel_prediction.cpu().numpy(),
            ],
            ["Synthetized Spectrogram"],
        )
        plt.savefig(fig_save_dir)
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[0].transpose(1, 2)
    lengths = predictions[11] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}{}.wav".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.wav".format(basename, teacher_forced_tag)),
            sampling_rate, wav)


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, figsize=(8, len(data) * 4), squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig

def plot_multi_attn(data, titles=None, save_dir=None):
    figs = list()
    for i, attn in enumerate(data):
        fig = plt.figure()
        num_head = attn.shape[0]
        for j, head_ali in enumerate(attn):
            ax = fig.add_subplot(2, num_head // 2, j + 1)
            ax.set_xlabel(
                'Audio timestep') if j % 2 == 1 else None
            ax.set_ylabel('Text timestep') if j >= num_head-2 else None
            im = ax.imshow(head_ali, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        figs.append(fig)
        if save_dir is not None:
            plt.savefig(save_dir[i])
        plt.close()

    return figs


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r','b'
    labels = 'Female','Male'

    data_x = embedding
    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig

def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def word_level_pooling(src_seq, src_len, wb, src_w_len, reduce="sum"):
    """
    src_seq -- [batch_size, max_time, dim]
    src_len -- [batch_size,]
    wb -- [batch_size, max_time]
    src_w_len -- [batch_size,]
    """
    batch, device = [], src_seq.device
    for s, sl, w, wl in zip(src_seq, src_len, wb, src_w_len):
        m, split_size = s[:sl, :], list(w[:wl].int())
        m = nn.utils.rnn.pad_sequence(torch.split(m, split_size, dim=0))
        if reduce == "sum":
            m = torch.sum(m, dim=0)  # [src_w_len, hidden]
        elif reduce == "mean":
            m = torch.div(torch.sum(m, dim=0), torch.tensor(
                split_size, device=device).unsqueeze(-1))  # [src_w_len, hidden]
        else:
            raise ValueError()
        batch.append(m)
    return pad(batch).to(device)

def word_level_subdivision(phones_per_word, max_phoneme_num):
    res = []
    for l in phones_per_word:
        if l <= max_phoneme_num:
            res.append(l)
        else:
            s, r = l//max_phoneme_num, l % max_phoneme_num
            res += [max_phoneme_num]*s + ([r] if r else [])
    return res

def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_mode == "vpsde":
        schedule_list = np.array([
            vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
    else:
        raise NotImplementedError
    return schedule_list

def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window
        if window is None:
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)
