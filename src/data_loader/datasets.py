import torch
from torch.utils.data import Dataset
from src.data_loader.imagelist import *
import pickle

class CustomDatasetCaption(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split, clip_model):
        assert dataset in ["coco-lt", "voc-lt", "coco", "voc"]
        if dataset == 'coco-lt':
            self.data_source = ImageListCaption(root='/data/coco/data/',
                                        list_file='../CoPE-main/data/coco/coco_lt_%s.txt' % split,
                                        caption_file='../CoPE-main/data/coco/coco_lt_captions.txt',
                                        label_file='../CoPE-main/data/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split,
                                        clip_model=clip_model)
        elif dataset == 'voc-lt':
            # 已永久修改为你的 VOC 数据根目录
            self.data_source = ImageListCaption(root='../CoPE-main/data/',
                                        list_file='../CoPE-main/data/voc/voc_lt_%s.txt' % split,
                                        caption_file='../CoPE-main/data/voc/voc_lt_captions.txt',
                                        label_file='../CoPE-main/data/voc/voc_labels.txt',
                                        nb_classes=20,
                                        split=split,
                                        clip_model=clip_model)

        self.targets = self.data_source.labels # one-hot label
        self.captions = self.data_source.captions_tokenized
        self.captions_ = self.data_source.captions_embedding
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target, caption, caption_ = self.data_source.get_sample(idx)
        return img, target, caption, caption_ 

class CustomDataset(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split, inp_name=None):
        assert dataset in ["coco-lt", "voc-lt", "voc", "nus-wide"]
        if dataset == 'coco-lt':
            self.data_source = ImageList(root='../CoPE-main/data/',
                                        list_file='../CoPE-main/data/coco/coco_lt_%s.txt' % split,
                                        label_file='../CoPE-main/data/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split)
        elif dataset == 'voc-lt':
            # 永久修改：你的VOC根目录 + 去掉了别人的路径
            self.data_source = ImageList(root='../CoPE-main/data/',
                                        list_file='../CoPE-main/data/voc/voc_lt_%s.txt' % split,
                                        label_file='../CoPE-main/data/voc/voc_labels.txt',
                                        nb_classes=20,
                                        split=split)

        self.targets = self.data_source.labels # one-hot label
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        return img, target, self.inp

def build_dataset_with_caption(dataset, split, clip_model=None):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt"]
    if split == 'train':
        dataset = CustomDatasetCaption(
        dataset=dataset, 
        split=split,
        clip_model=clip_model
        )
    elif split == 'test':
        dataset = CustomDataset(
        dataset=dataset, 
        split=split
        )

    return dataset

def build_dataset(dataset, split, inp_name):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt"]
    if split == 'train':
        dataset = CustomDataset(
        dataset=dataset, 
        split=split,
        inp_name=inp_name
        )
    elif split == 'test':
        dataset = CustomDataset(
        dataset=dataset, 
        split=split,
        inp_name=inp_name
        )

    return dataset