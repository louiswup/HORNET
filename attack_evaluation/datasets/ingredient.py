from pathlib import Path
from typing import Optional

import numpy as np
from sacred import Ingredient
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from .imagenet import load_imagenet

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def config():
    root = 'data'
    num_samples = None  # number of samples to attack, None for all
    random_subset = True  # True for random subset. False for sequential data in Dataset
    batch_size = 128


@dataset_ingredient.capture
def get_mnist(root: str) -> Dataset:
    transform = transforms.ToTensor()
    dataset = MNIST(root=root, train=False, transform=transform, download=True)
    return dataset


@dataset_ingredient.capture
def get_cifar10(root: str) -> Dataset:
    transform = transforms.ToTensor()
    dataset = CIFAR10(root=root, train=False, transform=transform, download=True)
    return dataset


@dataset_ingredient.capture
def get_imagenet(root: str, num_samples: Optional[int] = None) -> Dataset:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    data_path = Path(root) / 'imagenet-data'
    dataset = load_imagenet(root=data_path, split='val', transform=transform, n_samples=num_samples)
    print("Dataset size: ", len(dataset))
    return dataset


_datasets = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'imagenet': get_imagenet,
}


@dataset_ingredient.capture
def get_dataset(dataset: str):
    return _datasets[dataset]()


@dataset_ingredient.capture
def get_loader(dataset: str, batch_size: int, num_samples: Optional[int] = None,
               random_subset: bool = True) -> DataLoader:
    data = get_dataset(dataset=dataset)

    if num_samples is not None and num_samples < len(data):
        if not random_subset:
            data = Subset(data, indices=list(range(num_samples)))
        else:
            indices = np.random.choice(len(data), replace=False, size=num_samples)
            data = Subset(data, indices=indices)
    loader = DataLoader(dataset=data, batch_size=batch_size)
    return loader

'''@dataset_ingredient.capture
def get_adv_loader(dataset: str, adv_data, label, batch_size: int, num_samples: Optional[int] = None,
               random_subset: bool = True) -> DataLoader:
    data = get_dataset(dataset=dataset)
    data.data = (adv_data*255).round().numpy().astype(np.uint8)
    data.targets = label.numpy().tolist()


    if num_samples is not None and num_samples < len(data):
        if not random_subset:
            data = Subset(data, indices=list(range(num_samples)))
        else:
            indices = np.random.choice(len(data), replace=False, size=num_samples)
            data = Subset(data, indices=indices)
    loader = DataLoader(dataset=data, batch_size=batch_size)
    return loader'''
