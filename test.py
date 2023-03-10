import os
from glob import iglob

import gluonnlp as nlp
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    random_split,
)
from tqdm.notebook import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


class SpectrumDataset(Dataset):
    def __init__(self, dataset, labels):

        self.spectrums = [i for i in dataset]
        self.labels = [i for i in labels]

    def __getitem__(self, i):
        return (self.spectrums[i], self.labels[i])

    def __len__(self):
        return len(self.labels)


class Model(nn.Module):
    def __init__(self, output_size=115, p=0.2):
        super(Model, self).__init__()

        self.hidden_size = 256
        self.batch_size = 64
        self.num_layers = 10
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.4,
        )

        self.layers4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, output_size),
            nn.Softmax(),
        )

    def forward(self, X):
        h_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        )
        c_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        )
        output, hidden = self.lstm(X, (h_0, c_0))

        output = output[:, -1]  # 최종 예측 Hidden Layer
        output = self.layers4(output)
        return output


dataset = np.load("./sdataset.npy", allow_pickle=True)
labels = np.load("./labels.npy", allow_pickle=True)
d, l = dataset, labels


dataset, labels = torch.tensor(d, dtype=torch.float32), torch.tensor(
    l, dtype=torch.long
)
# dataset = np.transpose(dataset, [0, 2, 1])
labels = labels - 104
labels
new_labels = torch.zeros((len(labels), 115))
new_labels
for i, j in enumerate(labels):
    new_labels[i][int(j)] = 1
# dataset = dataset.view(21478, 1, 500, 40)
sdataset = SpectrumDataset(dataset, new_labels)
train_dataset, test_dataset = random_split(sdataset, [0.8, 0.2])
train_sampler = RandomSampler(train_dataset)
test_sampler = SequentialSampler(test_dataset)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler, batch_size=64, num_workers=0, drop_last=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, sampler=test_sampler, batch_size=64, num_workers=0, drop_last=True
)

device = torch.device("cuda")
model = Model(output_size=115, p=0.4).to(device)
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

batch_size = 64
warmup_ratio = 0.1
num_epochs = 300
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total
)


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    a, b = torch.max(Y, 1)
    train_acc = (max_indices == b).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


best_acc = 0
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (spectrum, label) in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):
        optimizer.zero_grad()
        spectrum = spectrum.to(device)
        label = label.to(device)

        out = model(spectrum)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print(
                "epoch {} batch id {} loss {} train acc {}".format(
                    e + 1,
                    batch_id + 1,
                    loss.data.cpu().numpy(),
                    train_acc / (batch_id + 1),
                )
            )
    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

    model.eval()  # 배치정규화,드랍아웃 같은 계층 off, 추론 시작
    with torch.no_grad():
        for batch_id, (spectrum, label) in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            spectrum = spectrum.to(device)
            label = label.to(device)
            out = model(spectrum)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
        if (test_acc / (batch_id + 1)) > best_acc:
            best_acc = test_acc / (batch_id + 1)

print("best_acc :", best_acc)
