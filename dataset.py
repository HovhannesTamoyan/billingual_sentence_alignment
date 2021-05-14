from typing import List, Dict
from datasets import load_dataset
import random


class Dataset():
    def __init__(self,
                 dataset_name: str):
        self._dataset = load_dataset(dataset_name, 'all_languages')['train']
        self._num_samples = len(self._dataset)

    def extract_lines(self,
                      num_lines: int,
                      langs: List[str],
                      seed: int):

        lines: Dict[str: List[str]] = dict()
        random.seed(seed)
        random_ids = random.sample(range(self._num_samples), num_lines)

        for lang in langs:
            lang_lines: List[str] = []
            for i in random_ids:
                lang_lines.append((i, self._dataset[i]['premise'][lang]))

            lines[lang] = lang_lines

        return lines
