from datasets import load_dataset
from safetensors.torch import save_file
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import typer

from pathlib import Path

def load_data():
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    special_token = tokenizer.all_special_tokens

    print(vocab_size, special_token)

    return dataset, tokenizer

def prepare(dataset_name, data_folder, context_length, dataset, tokenizer):

    tokens = []
    texts = []
    for sample in tqdm(dataset[dataset_name], dataset_name):
        tokens.extend(tokenizer(sample['text'])['input_ids'])
        if len(tokens) >= context_length:
            save, tokens = tokens[:context_length], tokens[context_length:]
            texts.append(save)
            
    texts = torch.tensor(texts)
    save_file({"text": texts}, Path(data_folder, F"{dataset_name}.safetensors"))

def main(data_folder: Path = "data/", context_length: int = 512):
    data_folder.mkdir(exist_ok=True)
    dataset, tokenizer = load_data()
    prepare("test", data_folder, context_length, dataset, tokenizer)
    prepare("validation", data_folder, context_length, dataset, tokenizer)
    prepare("train", data_folder, context_length, dataset, tokenizer)

if __name__ == "__main__":
    typer.run(main)