from typing import List
import numpy as np
from tokenizers import Tokenizer


class Topic:
    def __init__(self, topic_path: str, feature_size: int, topic_num: int, **kwargs):
        self.topic = CachedTopic(topic_path)
        self.feature_size = feature_size  # maxseq
        self.topic_num = topic_num

    def __call__(self, tokenizer: Tokenizer, words: List[List[str]]) -> np.ndarray:
        topic_features = []
        for word in words:
            id_list = [tokenizer.token_to_id(token) for token in word]  # 词片
            if len(id_list) < 5:
                id_list.extend([0] * (5 - len(id_list)))
            feature = self.topic(tuple(id_list))  # 全词
            topic_features.append(feature)  # 文档
        topic_features = np.vstack(topic_features).astype(np.float32)
        return topic_features

class CachedTopic:
    def __init__(self, path: str):
        self.cached_topic = np.load(path, allow_pickle=True).item()

    def __call__(self, token_ids: tuple) -> np.ndarray:
        if token_ids not in self.cached_topic:
            return np.zeros(64)
        return self.cached_topic[token_ids]