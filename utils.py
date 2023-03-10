import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    def __init__(self, dataset, labels):

        self.spectrums = [i for i in dataset]
        self.labels = [i for i in labels]

    def __getitem__(self, i):
        return (self.spectrums[i], self.labels[i])

    def __len__(self):
        return len(self.labels)


def load_dataset(padding=1):
    if padding == 0:
        dataset = np.load("./padding_dataset.npy")
        labels = np.load("./padding_labels.npy")
    # dataset = dataset.view(21478, 1, 500, 40)
    else:
        dataset = np.load("./no_padding_dataset.npy")
        labels = np.load("./no_padding_labels.npy")
        dataset = dataset.astype(np.float32)
        dataset = dataset.transpose(0, 2, 1)
    return dataset, labels


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc
