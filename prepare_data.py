from datasets import load_dataset
from safetensors.torch import save_file
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from pathlib import Path


CTX_LEN = 512
data_folder = Path("data/")


data_folder.mkdir(exist_ok=True)

dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
special_token = tokenizer.all_special_tokens

print(vocab_size, special_token)

def prepare(dataset_name):
    tokens = []
    texts = []
    for sample in tqdm(dataset[dataset_name], dataset_name):
        tokens.extend(tokenizer(sample['text'])['input_ids'])
        if len(tokens) >= CTX_LEN:
            save, tokens = tokens[:CTX_LEN], tokens[CTX_LEN:]
            texts.append(save)
            
    texts = torch.tensor(texts)
    save_file({"text": texts}, Path(data_folder, F"{dataset_name}.safetensors"))

prepare("test")
prepare("validation")
prepare("train")
