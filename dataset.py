import re
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from tokenizers.implementations import BertWordPieceTokenizer
from Topic import Topic
from datasets import load_dataset
from omegaconf.dictconfig import DictConfig
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from typing import Any, Dict, List
from projection import Projection
import json

class TPMixerDataModule(LightningDataModule):
    def __init__(self, vocab_cfg: DictConfig, train_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(TPMixerDataModule, self).__init__()
        self.vocab_cfg = vocab_cfg
        self.train_cfg = train_cfg
        self.topic_cfg = model_cfg.topic
        self.pro_cfg = model_cfg.projection
        self.topic = Topic(vocab_cfg.vocab_path, self.train_cfg.max_seq_len, self.topic_cfg.topic_num)
        self.tokenizer = BertWordPieceTokenizer(**self.vocab_cfg.tokenizer)
        self.projection = Projection(self.vocab_cfg.vocab_path2,
                                    self.pro_cfg.feature_size)

    def get_dataset_cls(self):
        if self.train_cfg.dataset_type == 'AGDataset':
            return AGDataset
        if self.train_cfg.dataset_type == 'ImdbDataset':
            return ImdbDataset
        if self.train_cfg.dataset_type == 'SST2Dataset':
            return SST2Dataset
        if self.train_cfg.dataset_type == 'ColaDataset':
            return CoLADataset
        if self.train_cfg.dataset_type == 'MTOPDataset':
            return MtopDataset

    def setup(self, stage: str = None):
        label_list = Path(self.train_cfg.labels).read_text().splitlines() if isinstance(self.train_cfg.labels,
                                                                                        str) else self.train_cfg.labels
        self.label_map = {label: index for index, label in enumerate(label_list)}
        dataset_cls = self.get_dataset_cls()

        if stage in (None, 'fit'):
            self.train_set = dataset_cls('train', self.train_cfg.max_seq_len, self.tokenizer,
                                         self.topic, self.projection, self.label_map, self.train_cfg.type)
            if self.train_cfg.dataset_type in ['ImdbDataset', 'AGDataset', 'MTOPDataset']:
                mode = 'test'
            else:
                mode = 'validation'
            self.eval_set = dataset_cls(mode, self.train_cfg.max_seq_len, self.tokenizer, self.topic,self.projection,
                                        self.label_map, self.train_cfg.type)
        if stage in (None, 'test'):
            if self.train_cfg.dataset_type in ['ImdbDataset', 'AGDataset', 'MTOPDataset']:
                mode = 'test'
            else:
                mode = 'validation'
            self.test_set = dataset_cls(mode, self.train_cfg.max_seq_len, self.tokenizer, self.topic,self.projection,
                                        self.label_map, self.train_cfg.type)

    # self.train_set: 表示训练集的数据集对象。
    # self.modelcfg.loader.batch_size: 表示每个批次的大小。
    # shuffle=True: 表示是否打乱数据。
    # num_workers=self.modelcfg.loader.num_workers: 表示读取数据时使用的线程数。
    # persistent_workers=True: 表示是否使用持久化工作线程。
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.train_cfg.train_batch_size, shuffle=True,
                          num_workers=self.train_cfg.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.train_cfg.test_batch_size, shuffle=False,
                          num_workers=self.train_cfg.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.train_cfg.test_batch_size, shuffle=False,
                          num_workers=self.train_cfg.num_workers, persistent_workers=True)


class TPMixerDataset(Dataset):
    def __init__(self, max_seq_len: int, tokenizer: Tokenizer, topic: Topic, projection: Projection,
                 label_map: Dict[str, int], name: str,
                 **kwargs):
        super(TPMixerDataset, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.topic = topic
        self.max_seq_len = max_seq_len
        self.label_map = label_map
        self.projection = projection
        self.name = name

    def normalize(self, text: str) -> str:
        return text.replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"')

    def topic_features(self, words: List[str]) -> np.ndarray:
        encoded = self.tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        tokens = [[] for _ in range(len(words))]
        for index, token in zip(encoded.words, encoded.tokens):
            tokens[index].append(token)
        features1 = self.topic(self.tokenizer, tokens)
        features2 = self.projection(tokens)
        padded_features1 = np.pad(features1, ((0, self.max_seq_len - len(words)), (0, 0)))
        padded_features2 = np.pad(features2, ((0, self.max_seq_len - len(words)), (0, 0)))
        return padded_features1, padded_features2

    def get_words(self, fields: List[str]) -> List[str]:
        raise NotImplementedError

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, index) -> Dict[str, Any]:
        if self.name == 'mtopold':
            fields = self.data[index].split('\t')
        else:
            fields = self.data[index]
        words = self.get_words(fields)
        features = self.topic_features(words)
        labels = self.compute_labels(fields)

        return {
            'inputs1': features[0],
            'inputs2': features[1],
            'targets': labels
        }


class AGDataset(TPMixerDataset):

    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(AGDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('ag_news', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class SST2Dataset(TPMixerDataset):

    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(SST2Dataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'sst2', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class CoLADataset(TPMixerDataset):

    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(CoLADataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'cola', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class ImdbDataset(TPMixerDataset):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        super(ImdbDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('imdb', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class MtopDataset(TPMixerDataset):
    def __init__(self, filename: str, *args, **kwargs):
        super(MtopDataset, self).__init__(*args, **kwargs)
        self.data = []
        root = Path('mtop/')
        # for file in root.glob(f'fr/{filename}.txt'):
        #     with open(file, 'r', encoding='utf-8') as f:
        #         self.data.extend(f.read().splitlines())

        for file in root.glob(f'de/{filename}.txt'):
            self.data.extend(file.read_text(encoding='utf-8').replace('，',',').splitlines())

    def __len__(self) -> int:
        return len(self.data)

    def get_words(self, fields: List[str]) -> List[str]:
        segments = json.loads(fields[-1].replace('"""','"\\""'))
        normalized_words = [self.normalize(word) for word in segments['tokens']]
        return normalized_words

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        segments = json.loads(fields[-1].replace('"""','"\\""'))
        num_words = len(segments['tokens'])
        slot_list = fields[2].split(',')
        slot = np.ones([num_words], dtype=np.int64) * self.label_map['O']
        slot = np.pad(slot, (0, self.max_seq_len - num_words), constant_values=-1)
        starts = {}
        ends = {}
        for index, span in enumerate(segments['tokenSpans']):
            starts[span['start']] = index
            ends[span['start'] + span['length']] = index + 1
        for s in slot_list:
            if not s:
                break
            start, end, _, val = s.split(':', maxsplit=3)
            # try:
            #     start_index = starts[int(start)]
            # except:
            #     print(fields)
            start_index = starts[int(start)]
            end_index = ends[int(end)]
            slot[start_index] = self.label_map[f'B-{val}']
            if end_index > start_index + 1:
                slot[start_index+1:end_index] = self.label_map[f'I-{val}']
        return slot