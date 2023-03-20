from safetensors import safe_open
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from pathlib import Path

from models.bind_rnn import BindRNN


batch_size = 128
accumulate_steps = 8
dims = 512
classes = 30522 #BertTokenizer
learning_rate = 0.01
device = torch.device("cuda")
epochs = 1
gradient_clipping = 1.0
data_folder = Path("data/")
num_workers = 4


class PreprocessedTextDataset(Dataset):
    def __init__(self, file_path: Path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            self.text = f.get_tensor("text")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        item = self.text[idx]
        text, label = item[:-1], item[1:]
        return text, label


trainset = PreprocessedTextDataset(Path(data_folder, "train.safetensors"))
trainloader = DataLoader(
    trainset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers, 
    drop_last=True
)

model = BindRNN(dims, dims * 4, classes).to(device)
model = nn.DataParallel(model)
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

run = wandb.init(project="BindRNN")
accumulate_counter, running_loss, running_acc = 0, 0, 0
for epoch in range(epochs):
    for text, label in trainloader:
        #forward + backward
        text = text.to(device)
        label = label.to(device)
        loss, acc = model(text, label)
        loss = loss.mean()
        acc = acc.mean()
        loss.backward()
        running_loss += loss.item()
        running_acc += acc.item()

        #update parameters
        accumulate_counter += 1
        if accumulate_counter == accumulate_steps:
            accumulate_counter = 0
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optim.step()
            optim.zero_grad()

            loss = running_loss / accumulate_steps
            acc = running_acc * 100 / accumulate_steps
            run.log({"loss": loss, "accuracy": acc})
            running_loss, running_acc = 0, 0
