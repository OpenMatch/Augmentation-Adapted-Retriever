import json
import re
import os
import torch
import math
import numpy as np
import pickle
from torch.utils.data import Dataset
from utils import print_rank_0, save_rank_0
from tokenization_t5 import EncDecTokenizer
from .data_config import DATA_GROUP_CONFIG, DATA_CONFIG
import datasets
from promptsource.templates import TemplateCollection
from datasets import load_dataset
from .postprocess import OPTION_POST_FN

datasets.disable_caching()


class T0Dataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer: EncDecTokenizer,
        data_prompts,
        split,
        ratio=1,
        few_data_names=None,
        num=-1,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.data_prompts = data_prompts
        self.pad_id = tokenizer.pad_id
        self.split = split
        self.sample_num = num
        self.idx_size = 3
        self.few_data_names = few_data_names
        self.selfsup_sample_num = {"train": 100000, "validation": 1000}

        self.all_data = {name: {} for name in data_prompts}
        self.all_enc_sizes = []
        self.all_dec_sizes = []
        self.all_cand_sizes = []
        if self.args.FiD:
            self.all_passage_sizes = []
        for name in self.data_prompts:
            if DATA_CONFIG[name].get("selfsup", False):
                data, enc_sizes, dec_sizes, cand_sizes = self.load_from_cache_self(name)
                self.all_data[name] = {
                    "prompt_num": 1,
                    "prompt_names": ["merged"],
                    "data": data,
                }
            else:
                if DATA_CONFIG[name]["do_cache"]:
                    (
                        data,
                        enc_sizes,
                        dec_sizes,
                        cand_sizes,
                        passage_sizes,
                    ) = self.load_from_cache(name)
                else:
                    (
                        data,
                        enc_sizes,
                        dec_sizes,
                        cand_sizes,
                        passage_sizes,
                    ) = self.process_data(name)

                self.all_data[name] = {
                    "prompt_num": len(data_prompts[name]),
                    "prompt_names": [prompt.name for prompt in data_prompts[name]],
                    "data": data,
                }

            print("len data", len(data))

            self.all_enc_sizes.extend(enc_sizes)
            self.all_dec_sizes.extend(dec_sizes)
            self.all_cand_sizes.extend(cand_sizes)
            if self.args.FiD:
                self.all_passage_sizes.extend(passage_sizes)

        self.max_enc_len = max(self.all_enc_sizes)
        self.max_dec_len = max(self.all_dec_sizes)
        self.max_cand_len = max(self.all_cand_sizes)
        if self.args.FiD:
            self.max_passage_len = max(self.all_passage_sizes)
            self.max_enc_len = self.max_passage_len * self.args.passage_num

        self.flan_sample_num = {
            name: min(
                DATA_CONFIG[name].get("flan_sample_max", args.flan_sample_max)
                * d["prompt_num"],
                len(d["data"]),
            )
            for name, d in self.all_data.items()
        }
        self.idxs = self.build_idx()

        self.cur_epoch = 0

        print_str = ""
        for name in self.data_prompts:
            print_str += "Data: {}_{}".format(name, split)
            print_str += " | Ratio: {}".format(ratio)
            print_str += " | Max enc len: {}".format(self.max_enc_len)
            print_str += " | Max dec len: {}".format(self.max_dec_len)
            print_str += " | Max cand len: {}".format(self.max_cand_len)
            print_str += " | Prompt num: {}".format(self.all_data[name]["prompt_num"])
            print_str += " | All data num: {}".format(len(self.all_data[name]["data"]))
            print_str += " | Sample num: {}".format(self.flan_sample_num[name])
            print_str += " | Idx one epoch num: {}\n".format(len(self.idxs[0]))

        print_str = print_str[:-1]
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def set_epoch(self, e):
        self.cur_epoch = e

    def build_idx(self):
        epochs = self.args.epochs
        idx_repo = {}
        for (name, d), (name, sample_num) in zip(
            self.all_data.items(), self.flan_sample_num.items()
        ):
            data_idx = [i for i in range(len(d["data"]))]
            repeat_num = math.ceil(epochs * sample_num / len(data_idx))
            tmp_data_idx = []
            for i in range(repeat_num):
                if self.split == "train":
                    np.random.shuffle(data_idx)
                tmp_data_idx.extend(data_idx)
            idx_repo[name] = tmp_data_idx
            print(
                name,
                "| repeat num:",
                repeat_num,
                "| sample num:",
                sample_num,
                "| data_idx len:",
                len(data_idx),
                "| tmp_data_idx:",
                len(tmp_data_idx),
            )

        idxs = []
        for e in range(epochs):
            samp_idx = []
            for name, d in self.all_data.items():
                sample_num = self.flan_sample_num[name]
                l = idx_repo[name][e * sample_num : (e + 1) * sample_num]
                l = [(name, x) for x in l]
                samp_idx.extend(l)
            idxs.append(samp_idx)

        first_len = len(idxs[0])
        for e, x in enumerate(idxs):
            assert len(x) == first_len, (e, len(x), first_len)

        return idxs

    def load_from_cache_self(self, name):
        cache_path = os.path.join(
            DATA_CONFIG[name]["data_dir"],
            "cache_{}_{}.pkl".format(self.split, self.selfsup_sample_num[self.split]),
        )
        with open(cache_path, "rb") as f:
            data, enc_sizes, dec_sizes, cand_sizes = pickle.load(f)

        return data, enc_sizes, dec_sizes, cand_sizes

    def load_from_cache(self, name):
        data_dir = DATA_CONFIG[name]["data_dir"]

        if self.split == "train":
            if self.args.few_data_num is not None:
                assert self.few_data_names is not None
                if name in self.few_data_names:
                    sample_num = self.args.few_data_num
                else:
                    sample_num = self.sample_num
            else:
                sample_num = self.sample_num
            cache_path = os.path.join(
                data_dir,
                "cache_{}_{}_{}.pkl".format(self.split, self.ratio, sample_num),
            )
        else:
            prompt_name = self.data_prompts[name][0].name.replace("/", "_")
            cache_path = os.path.join(
                data_dir,
                "cache_{}_{}_{}_{}.pkl".format(
                    self.split, self.ratio, self.sample_num, prompt_name
                ),
            )

        print("cache path", cache_path)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data, enc_sizes, dec_sizes, cand_sizes, passage_sizes = pickle.load(f)
        else:
            data, enc_sizes, dec_sizes, cand_sizes, passage_sizes = self.process_data(
                name
            )
            with open(cache_path, "wb") as f:
                pickle.dump((data, enc_sizes, dec_sizes, cand_sizes, passage_sizes), f)

        return data, enc_sizes, dec_sizes, cand_sizes, passage_sizes

    def process_data(self, name):
        print_rank_0("Processing " + name)
        if self.split == "train":
            if self.args.few_data_num is not None:
                assert self.few_data_names is not None
                if name in self.few_data_names:
                    sample_num = self.args.few_data_num
                else:
                    sample_num = self.sample_num
            else:
                sample_num = DATA_CONFIG[name].get("train_num", self.sample_num)
                if self.args.data_aug is not None:
                    sample_num += self.args.data_aug
        else:
            sample_num = DATA_CONFIG[name].get("dev_num", self.sample_num)

        data_dir = DATA_CONFIG[name]["data_dir"]
        data_files = {self.split: os.path.join(data_dir, "{}.jsonl".format(self.split))}
        dataset = load_dataset("json", data_files=data_files)

        data = []
        enc_sizes = []
        dec_sizes = []
        cand_sizes = []
        passage_sizes = []
        sid, lid = 0, 0
        skip = 0
        for pid, prompt in enumerate(self.data_prompts[name]):
            print_rank_0(prompt.name)
            for sample in dataset[self.split]:
                if lid % 500 == 0:
                    print_rank_0(
                        "{}, {}, {}, {}, {}".format(
                            name, self.split, prompt.name, lid, skip
                        )
                    )

                # genread_template = "{} Generate a background document from Wikipedia to help answer the given question:"
                answers = None
                if "popQA" in name:
                    enc_str = sample["prompt"]
                    # enc_str = genread_template.format(enc_str)
                    dec_str = sample["answers"][0]
                    answers = sample["answers"]
                else:
                    applied_sample = prompt.apply(sample)
                    if len(applied_sample) != 2:
                        # print_rank_0("sample num out")
                        skip += 1
                        continue
                    enc_str, dec_str = applied_sample
                    # enc_str = genread_template.format(enc_str)

                if "mmlu_demo" in sample:
                    enc_str = sample["mmlu_demo"] + enc_str

                passages = None
                if "passages" in sample:
                    passages = []
                    for i in range(self.args.passage_num):
                        max_question_len = 1250 if self.split == "train" else 10000
                        max_passage_len = (
                            max(1250 - len(enc_str), 0)
                            if self.split == "train"
                            else 500
                        )
                        # Can last
                        if self.args.prompt_tune:
                            passage_str = enc_str[:max_question_len]
                            passages.append(
                                [-(i + 1)] + self.tokenizer.encode(passage_str) + [1]
                            )
                        else:
                            passage_str = (
                                sample["passages"][i][:max_passage_len]
                                + enc_str[:max_question_len]
                            )
                            passages.append(self.tokenizer.encode(passage_str) + [1])

                if self.args.prompt_tune:
                    context = (
                        [-(i + 1) for i in range(self.args.passage_num)]
                        + self.tokenizer.encode(enc_str)
                        + [1]
                    )
                else:
                    context = self.tokenizer.encode(enc_str) + [1]
                target = [0] + self.tokenizer.encode(dec_str) + [1]

                # if len(enc_str) > 5000:
                #     # print_rank_0("pre-check out " + str(len(enc_str)))
                #     skip += 1
                #     continue

                # if len(context) > self.args.enc_seq_length:
                #     skip += 1
                #     # print_rank_0("enc out " + str(len(context)))
                #     continue

                # if len(target) > self.args.dec_seq_length:
                #     skip += 1
                #     # print_rank_0("dec out " + str(len(target)))
                #     continue

                options = prompt.get_answer_choices_list(sample)
                options = OPTION_POST_FN.get((name, prompt.name), lambda x: x)(options)

                if self.split != "train" and options is not None:
                    cands = [
                        [0] + self.tokenizer.encode(option) + [1] for option in options
                    ]
                else:
                    cands = None

                if len(dec_str) == 0:
                    # print_rank_0("dec str out " + str(len(dec_str)))
                    skip += 1
                    continue

                if options is not None and dec_str not in options:
                    print_rank_0(str(applied_sample))
                    print_rank_0(
                        name
                        + " "
                        + prompt.name
                        + " "
                        + "Skip bug sample "
                        + str(dec_str)
                        + " "
                        + str(options)
                    )
                    continue

                data.append(
                    {
                        "idxs": [pid, lid, sid],
                        "enc_input_ids": context,
                        "dec_input_ids": target[:-1],
                        "label_ids": target[1:],
                        "answer": dec_str if answers is None else answers,
                        "options": options,
                        "cands": {
                            "input_ids": [c[:-1] for c in cands],
                            "target_ids": [c[1:] for c in cands],
                            "label": options.index(dec_str),
                        }
                        if cands is not None
                        else None,
                        "passage_input_ids": passages,
                    }
                )

                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)
                if cands is not None:
                    cand_sizes.append(sum([len(c) - 1 for c in cands]))
                else:
                    cand_sizes.append(0)
                if passages is not None:
                    passage_sizes.extend([len(p) for p in passages])
                else:
                    passage_sizes.append(0)

                sid += 1
                lid += 1
                if sample_num > 0 and lid >= sample_num:
                    break

            lid = 0

        return data, enc_sizes, dec_sizes, cand_sizes, passage_sizes

    def __len__(self):
        return len(self.idxs[0])

    def __getitem__(self, idx):
        name, sid = self.idxs[self.cur_epoch][idx]
        d = self.all_data[name]["data"][sid]
        return d, name

    def collate(self, samples):
        bs = len(samples)
        model_data = {
            "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long)
            * self.pad_id,
            "enc_attention_mask": torch.zeros(
                bs, 1, self.max_enc_len, self.max_enc_len
            ),
            "dec_attention_mask": torch.zeros(
                bs, 1, self.max_dec_len, self.max_dec_len
            ),
            "cross_attention_mask": torch.zeros(
                bs, 1, self.max_dec_len, self.max_enc_len
            ),
            "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long)
            * self.pad_id,
        }
        if self.args.FiD:
            model_data["passage_input_ids"] = (
                torch.ones(
                    bs, self.args.passage_num, self.max_passage_len, dtype=torch.long
                )
                * self.pad_id
            )
            model_data["passage_attention_mask"] = torch.zeros(
                bs, self.args.passage_num, 1, self.max_passage_len, self.max_passage_len
            )
        no_model_data = {
            "idxs": torch.zeros(bs, self.idx_size, dtype=torch.long),
            "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            "loss_mask": torch.zeros(bs, self.max_dec_len),
        }

        name_list = []
        for i, samp in enumerate(samples):
            samp, name = samp
            name_list.append(name)
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(
                samp["enc_input_ids"], dtype=torch.long
            )
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = samp.get(
                "enc_attention_mask", 1.0
            )
            model_data["dec_input_ids"][i][:dec_len] = torch.tensor(
                samp["dec_input_ids"], dtype=torch.long
            )
            model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(
                torch.ones(dec_len, dec_len)
            )
            if self.args.FiD:
                enc_len = self.max_enc_len
                samp["cross_attention_mask"] = torch.zeros(enc_len)
                for j in range(self.args.passage_num):
                    passage_len = len(samp["passage_input_ids"][j])
                    samp["cross_attention_mask"][
                        j * self.max_passage_len : j * self.max_passage_len
                        + passage_len
                    ] = 1.0
            model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = samp.get(
                "cross_attention_mask", 1.0
            )
            if self.args.FiD:
                for j in range(self.args.passage_num):
                    passage_len = len(samp["passage_input_ids"][j])
                    model_data["passage_input_ids"][i][j][:passage_len] = torch.tensor(
                        samp["passage_input_ids"][j], dtype=torch.long
                    )
                    model_data["passage_attention_mask"][i][j][
                        0, :passage_len, :passage_len
                    ] = 1.0
            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)

            no_model_data["labels"][i][: len(samp["label_ids"])] = torch.tensor(
                samp["label_ids"], dtype=torch.long
            )
            no_model_data["loss_mask"][i][: len(samp["label_ids"])] = 1.0

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data[
                "cross_attention_mask"
            ].half()
            if self.args.FiD:
                model_data["passage_attention_mask"] = model_data[
                    "passage_attention_mask"
                ].half()

        if samp["cands"] is not None:
            cand_model_data = {
                "dec_input_ids": torch.ones(bs, self.max_cand_len, dtype=torch.long)
                * self.pad_id,
                "dec_attention_mask": torch.zeros(
                    bs, 1, self.max_cand_len, self.max_cand_len
                ),
                "cross_attention_mask": torch.zeros(
                    bs, 1, self.max_cand_len, self.max_enc_len
                ),
            }
            cand_no_model_data = {
                "labels": torch.zeros(bs, dtype=torch.long),
                "target_ids": torch.ones(bs, self.max_cand_len, dtype=torch.long)
                * self.pad_id,
                "pos": torch.zeros(bs, self.max_cand_len, dtype=torch.bool),
                "loss_mask": torch.zeros(bs, self.max_cand_len),
            }

            for i, samp in enumerate(samples):
                samp, _ = samp
                start = 0
                enc_len = len(samp["enc_input_ids"])
                if self.args.FiD:
                    enc_len = self.max_enc_len

                for input_ids, target_ids in zip(
                    samp["cands"]["input_ids"], samp["cands"]["target_ids"]
                ):
                    cand_model_data["dec_input_ids"][i][
                        start : start + len(input_ids)
                    ] = torch.tensor(input_ids, dtype=torch.long)
                    cand_no_model_data["target_ids"][i][
                        start : start + len(target_ids)
                    ] = torch.tensor(target_ids, dtype=torch.long)
                    cand_model_data["dec_attention_mask"][i][
                        0,
                        start : start + len(input_ids),
                        start : start + len(input_ids),
                    ] = torch.tril(torch.ones(len(input_ids), len(input_ids)))
                    cand_model_data["cross_attention_mask"][i][
                        0, start : start + len(input_ids), :enc_len
                    ] = samp.get("cross_attention_mask", 1.0)
                    cand_no_model_data["loss_mask"][i][
                        start : start + len(input_ids)
                    ] = 1
                    start = start + len(input_ids)
                    cand_no_model_data["pos"][i][start - 1] = True
                cand_no_model_data["labels"][i] = samp["cands"]["label"]

            if self.args.fp16:
                cand_model_data["dec_attention_mask"] = cand_model_data[
                    "dec_attention_mask"
                ].half()
                cand_model_data["cross_attention_mask"] = cand_model_data[
                    "cross_attention_mask"
                ].half()
        else:
            cand_model_data, cand_no_model_data = {}, {}

        # print(name_list)

        return model_data, no_model_data, cand_model_data, cand_no_model_data
