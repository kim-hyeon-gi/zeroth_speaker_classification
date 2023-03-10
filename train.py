import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    random_split,
)
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from model import (
    conv1D_Model,
    conv2D_Model,
    lstm_Model,
    mean_Model,
    model_list,
    transformer_Model,
)
from utils import SpectrumDataset, calc_accuracy, load_dataset


def train_start(model):
    # set arg
    batch_size = 64
    train_rate = 0.8
    test_rate = 1 - train_rate
    warmup_ratio = 0.1
    num_epochs = 500
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5

    # data load
    dataset, labels = load_dataset(1)
    if isinstance(model, conv2D_Model):
        print(num_epochs)
    elif isinstance(model, lstm_Model):
        dataset = dataset.transpose(0, 2, 1)
    elif isinstance(model, transformer_Model):
        dataset = dataset.transpose(0, 2, 1)

    sdataset = SpectrumDataset(dataset, labels)
    train_dataset, test_dataset = random_split(sdataset, [train_rate, test_rate])
    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
    )

    # model load
    device = torch.device("cuda")
    model = model.to(device)

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
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total
    )

    best_acc = 0
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (spectrum, label) in enumerate(train_dataloader):
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
            for batch_id, (spectrum, label) in enumerate(test_dataloader):
                spectrum = spectrum.to(device)
                label = label.to(device)
                out = model(spectrum)
                test_acc += calc_accuracy(out, label)
            print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
            if (test_acc / (batch_id + 1)) > best_acc:
                best_acc = test_acc / (batch_id + 1)

    print("model :\n", model, "\nbest_acc :", best_acc)


train_start(model_list()[2])
