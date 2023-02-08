# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import logging
import os
import random
from typing import Callable, Dict, List, Union

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer

from ..arguments import DataArguments, DRPretrainingDataArguments
from ..data_augmentation_strategy import Cropping, NullStrategy, SequentialStrategies
from ..trainer import DRTrainer

logger = logging.getLogger(__name__)


class TrainDatasetBase:
    """
    Abstract base class for all train datasets in Openmatch.\n
    This implants arguments and data preparation, but should be mostly used for identifying an OpenMatch Train Dataset.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer
        self.is_eval = is_eval
        self._prepare_data(data_args, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        if not self.is_eval:
            self.data_files = (
                [data_args.train_path]
                if data_args.train_dir is None
                else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
            )
        else:
            self.data_files = [data_args.eval_path]

    def get_process_fn(self, epoch, hashed_seed):
        raise NotImplementedError


class StreamTrainDatasetMixin(IterableDataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir
        )["train"]
        self.dataset = (
            self.dataset.shuffle(seed=shuffle_seed, buffer_size=10_000)
            if shuffle_seed is not None
            else self.dataset
        )
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def __len__(self):
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count

    def __iter__(self):
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch)
            _hashed_seed = hash(self.trainer.args.seed)
            self.dataset.set_epoch(epoch)
            return iter(
                self.dataset.map(
                    self.get_process_fn(epoch, _hashed_seed),
                    remove_columns=self.all_columns,
                )
            )
        return iter(
            self.dataset.map(
                self.get_process_fn(0, None), remove_columns=self.all_columns
            )
        )


class MappingTrainDatasetMixin(Dataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False, cache_dir=cache_dir
        )["train"]
        sample = self.dataset[0]
        self.all_columns = sample.keys()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch)
            _hashed_seed = hash(index + self.trainer.args.seed)
            return self.get_process_fn(epoch, _hashed_seed)(group)
        return self.get_process_fn(0, None)(group)


class DRTrainDataset(TrainDatasetBase):
    def create_one_example(
        self, text_encoding: List[int], is_query=False
    ) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len
            if is_query
            else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            encoded_query = self.create_one_example(qry, is_query=True)
            encoded_passages = []
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            if not self.data_args.use_all_positive_passages:
                if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                    pos_psg = group_positives[0]
                else:
                    pos_psg = group_positives[
                        (hashed_seed + epoch) % len(group_positives)
                    ]
                encoded_passages.append(self.create_one_example(pos_psg))
            else:
                for pos_psg in group_positives:
                    encoded_passages.append(self.create_one_example(pos_psg))

            negative_size = self.data_args.train_n_passages - 1
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 2
                    negs = negs[:negative_size]
            elif self.data_args.train_n_passages == 1:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset : _offset + negative_size]

            for neg_psg in negs:
                encoded_passages.append(self.create_one_example(neg_psg))

            if not self.data_args.use_all_positive_passages:
                assert len(encoded_passages) == self.data_args.train_n_passages
            else:
                assert (
                    len(encoded_passages)
                    == self.data_args.train_n_passages + len(group_positives) - 1
                )

            return {
                "query_": encoded_query,
                "passages": encoded_passages,
            }  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamDRTrainDataset(StreamTrainDatasetMixin, DRTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, DRTrainDataset):
    pass


class DRPretrainDataset(TrainDatasetBase):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DRPretrainingDataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None,
    ) -> None:
        super(DRPretrainDataset, self).__init__(
            tokenizer, data_args, trainer, is_eval, shuffle_seed, cache_dir
        )
        pretrain_strategies_str = (
            data_args.pretrain_strategies.split(",")
            if data_args.pretrain_strategies is not None
            else []
        )
        strategies = []
        for strategy_str in pretrain_strategies_str:
            if strategy_str == "null":
                strategies.append(NullStrategy())
                logger.info("Adding NullStrategy")
            elif strategy_str == "crop":
                strategies.append(
                    Cropping(
                        ratio_min=data_args.cropping_ratio_min,
                        ratio_max=data_args.cropping_ratio_max,
                    )
                )
                logger.info(
                    "Adding Cropping, ratio_min={}, ratio_max={}".format(
                        data_args.cropping_ratio_min, data_args.cropping_ratio_max
                    )
                )
            else:
                raise ValueError(
                    "Unknown pretraining strategy: {}".format(strategy_str)
                )
        self.apply_strategy = SequentialStrategies(*strategies)

    def create_one_example(
        self, text_encoding: List[int], is_query=False
    ) -> BatchEncoding:
        text_encoding = self.apply_strategy(text_encoding)
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len
            if is_query
            else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            content = example[self.data_args.pretrain_target_field]
            encoded_query = self.create_one_example(content, is_query=True)
            encoded_passages = [self.create_one_example(content)]

            return {"query": encoded_query, "passages": encoded_passages}

        return process_fn


class StreamDRPretrainDataset(StreamTrainDatasetMixin, DRPretrainDataset):
    pass


class MappingDRPretrainDataset(MappingTrainDatasetMixin, DRPretrainDataset):
    pass


class RRTrainDataset(TrainDatasetBase):
    def create_one_example(self, qry_encoding, psg_encoding) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            qry_encoding + psg_encoding,
            truncation="longest_first",
            max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_pos_pair = self.create_one_example(qry, pos_psg)

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(hashed_seed + epoch) % len(group_negatives)]
            encoded_neg_pair = self.create_one_example(qry, neg_psg)
            return {"pos_pair": encoded_pos_pair, "neg_pair": encoded_neg_pair}

        return process_fn


class StreamRRTrainDataset(StreamTrainDatasetMixin, RRTrainDataset):
    pass


class MappingRRTrainDataset(MappingTrainDatasetMixin, RRTrainDataset):
    pass
