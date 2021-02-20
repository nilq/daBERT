from os.path import dirname, join
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

model_name = 'dabert'
file_dir   = dirname(__file__)

data_folder = join(file_dir, 'data')
data_paths  = [str(x) for x in Path(data_folder).glob("**/*.txt")]

output_folder = join(file_dir, f'models/{model_name}/')

vocab_file    = join(output_folder, f'vocab.json')
merges_file   = join(output_folder, f'merges.txt')

if __name__ == "__main__":
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=data_paths,
        vocab_size=52000,
        min_frequency=2,
        special_tokens = [
            '<s>',
            '<pad>',
            '</s>',
            '<unk>',
            '<mask>',
        ]
    )

    tokenizer.save_model(output_folder, model_name)
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )