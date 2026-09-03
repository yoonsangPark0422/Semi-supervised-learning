import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    train_pool_idxs, val_idxs = stratified_train_val_split(
        base_dataset.targets, args)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets, train_pool_idxs)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    test_dataset = CIFAR10SSL(
        root, val_idxs, train=True, transform=transform_val, download=False)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    base_dataset = datasets.CIFAR100(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))
    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_svhn(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomCrop(size=32, padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    base_dataset = datasets.SVHN(root, split='train', download=True)
    train_pool_idxs, val_idxs = stratified_train_val_split(
        base_dataset.labels, args)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.labels, train_pool_idxs)

    train_labeled_dataset = SVHNSSL(
        root, train_labeled_idxs, split='train', transform=transform_labeled)
    train_unlabeled_dataset = SVHNSSL(
        root, train_unlabeled_idxs, split='train',
        transform=TransformFixMatch(mean=svhn_mean, std=svhn_std))
    test_dataset = SVHNSSL(
        root, val_idxs, split='train', transform=transform_val,
        download=False)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def stratified_train_val_split(labels, args):
    labels = np.asarray(labels)
    val_per_class = int(getattr(args, 'val_samples_per_class', 500))
    rng = np.random.RandomState(args.seed if args.seed is not None else 0)
    train_idx = []
    val_idx = []
    for class_idx in range(args.num_classes):
        idx = np.where(labels == class_idx)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        val_idx.extend(idx[:val_per_class])
        train_idx.extend(idx[val_per_class:])
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    logger.info('Using %d validation samples per class from train split.',
                val_per_class)
    return train_idx, val_idx


def x_u_split(args, labels, candidate_idxs=None):
    labels = np.array(labels)
    if candidate_idxs is None:
        candidate_idxs = np.array(range(len(labels)))
    else:
        candidate_idxs = np.asarray(candidate_idxs)
    labeled_idx = []
    unlabeled_idx = candidate_idxs.copy()
    candidate_labels = labels[candidate_idxs]
    if getattr(args, 'imb_ratio', 1.0) > 1.0:
        img_num_per_cls = make_imb_data(args.num_labeled, args.num_classes,
                                        args.imb_ratio, args.imb_type)
    else:
        img_num_per_cls = [args.num_labeled // args.num_classes] * args.num_classes

    for i in range(args.num_classes):
        idx = candidate_idxs[np.where(candidate_labels == i)[0]]
        idx = np.random.choice(idx, img_num_per_cls[i], False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)
    args.labeled_class_counts = img_num_per_cls
    args.num_labeled = len(labeled_idx)
    assert args.num_labeled == sum(img_num_per_cls)
    logger.info('Labeled class counts: %s', img_num_per_cls)

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def make_imb_data(max_num, class_num, gamma, imb_type):
    if imb_type == 'step':
        ratios = [1.0] * (class_num // 2) + \
            [1.0 / gamma] * (class_num - class_num // 2)
    else:
        ratios = [math.pow(1.0 / gamma, cls_idx / (class_num - 1.0))
                  for cls_idx in range(class_num)]

    unit = max_num / sum(ratios)
    img_num_per_cls = [max(1, int(unit * ratio)) for ratio in ratios]
    diff = max_num - sum(img_num_per_cls)
    order = list(range(class_num)) if diff >= 0 else list(reversed(range(class_num)))
    while diff != 0:
        changed = False
        for cls_idx in order:
            if diff > 0:
                img_num_per_cls[cls_idx] += 1
                diff -= 1
            elif img_num_per_cls[cls_idx] > 1:
                img_num_per_cls[cls_idx] -= 1
                diff += 1
            changed = True
            if diff == 0:
                break
        if not changed:
            break
    return img_num_per_cls


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, transform=transform,
                         target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.targets = self.labels

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'svhn': get_svhn}
