import torch

from torch.utils.data import Dataset
from tokenizer import vocab_file, merges_file, data_folder
from pathlib import Path
from os.path import join, dirname
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

class DanishDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            vocab_file,
            merges_file
        )

        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        src_files = Path(data_folder).glob("**/*.txt")

        for src_file in src_files:
            print("🇩🇰", src_file)

            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])