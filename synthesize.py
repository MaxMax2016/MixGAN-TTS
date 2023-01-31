import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import json
import argparse
from string import punctuation

from pypinyin import pinyin, Style
from tqdm import tqdm

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples, word_level_subdivision
from dataset import Dataset, TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)             # 把尾部的符号去掉
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()                                                     # 英文字素转音素模块
    phones = []                                                     # 转换后的音素
    word_boundaries = []                                            # 词边界
    words = re.split(r"([,;.\-\?\!\s+])", text)                     # 用这些符号进行分词切割
    for w in words:
        if w.lower() in lexicon:                                    # 正常英文  --  lower 将所有的大写改成小写
            phone_list = lexicon[w.lower()]
        else:                                                       # 标点符号  -- 分隔符
            phone_list = list(filter(lambda p: p != " ", g2p(w)))   # p不是英文字符  也不是空格符   特殊符号
        if phone_list:
            phones += phone_list
            word_boundaries.append(len(phone_list))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    if preprocess_config["preprocessing"]["text"]["sub_divide_word"]:
        word_boundaries = word_level_subdivision(
            word_boundaries, preprocess_config["preprocessing"]["text"]["max_phoneme_num"])

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(
        phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    ))

    return np.array(sequence), np.array(word_boundaries)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    word_boundaries = []
    phones = []
    phone_list = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:                       # pinyins为中文音标带音调   ['ni3', 'hao3', 'a5', 'shi4', 'jie4']
        if p in lexicon:
            phone_list = (lexicon[p])
        else:
            phone_list.append("sp")
        if phone_list:
            phones += phone_list
            word_boundaries.append(len(phone_list))

    phones = "{" + " ".join(phones) + "}"
    if preprocess_config["preprocessing"]["text"]["sub_divide_word"]:
            word_boundaries = word_level_subdivision(
                word_boundaries, preprocess_config["preprocessing"]["text"]["max_phoneme_num"]
            )

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        ))

    return np.array(sequence), np.array(word_boundaries)


def synthesize(model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    def synthesize_(batch):
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-1]),
                spker_embeds=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )[0]
            synth_samples(
                args,
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                model.diffusion,
            )

    if args.teacher_forced:
        for batchs_ in batchs:
            for batch in tqdm(batchs_):
                batch = list(batch)
                batch[6] = None # set mel None for diffusion sampling
                synthesize_(batch)
    else:
        for batch in tqdm(batchs):
            synthesize_(batch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
        help="training model type",
    )
    parser.add_argument("--teacher_forced", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default="42",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.text is None
        if args.teacher_forced:
            assert args.source is None
        else:
            assert args.source is not None
    if args.mode == "single":
        assert args.source is None and args.text is not None and not args.teacher_forced

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    if args.model == "shallow":
        assert args.restore_step >= train_config["step"]["total_step_aux"]
    if args.model in ["aux", "shallow"]:
        train_tag = "shallow"
    elif args.model == "naive":
        train_tag = "naive"
    else:
        raise NotImplementedError
    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"_{}{}".format(args.model, path_tag)
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Log Configuration
    print("\n==================================== Inference Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        if args.teacher_forced:
            dataset = Dataset(
                "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
            )
        else:
            dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array(
            [0])  # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts, word_boundaries = preprocess_english(
                args.text, preprocess_config)
            texts, word_boundaries = np.array(
                [texts]), np.array([word_boundaries])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts, word_boundaries = preprocess_mandarin(
                args.text, preprocess_config)
            texts, word_boundaries = np.array(
                [texts]), np.array([word_boundaries])

        text_lens = np.array([len(texts[0])])
        text_w_lens = np.array([len(word_boundaries[0])])
        batchs = [(ids, raw_texts, speakers, texts,
                   text_lens, max(text_lens), word_boundaries, text_w_lens, max(text_w_lens), spker_embed)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args, configs, vocoder, batchs, control_values)
