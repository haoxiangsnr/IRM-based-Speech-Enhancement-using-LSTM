from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """
    定义测试集
    """

    def __init__(self,
                 mixture_dataset="/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/DNN/pad_3_400_50/test/mixture.npy",
                 clean_dataset="/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/DNN/pad_3_400_50/test/clean.npy",
                 limit=None,
                 offset=0):
        """
        构建测试数据集

        Args:
            mixture_dataset (str): 带噪语音数据集
            clean_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        mixture_dataset = Path(mixture_dataset)
        clean_dataset = Path(clean_dataset)

        assert mixture_dataset.exists() and clean_dataset.exists(), "测试数据集不存在"

        print(f"Loading mixture dataset {mixture_dataset.as_posix()} ...")
        mixture_dataset_dict: dict = np.load(mixture_dataset.as_posix()).item()
        print(f"Loading clean dataset {clean_dataset.as_posix()} ...")
        clean_dataset_dict: dict = np.load(clean_dataset.as_posix()).item()
        assert len(mixture_dataset_dict) % len(clean_dataset_dict) == 0, "mixture dataset 的长度不是 clean dataset 的整数倍"

        print(f"The len of fully dataset is {len(mixture_dataset_dict)}.")
        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")

        if limit is None:
            limit = len(mixture_dataset_dict)

        self.keys = list(mixture_dataset_dict.keys())
        self.keys.sort()
        self.keys = self.keys[offset: offset + limit]

        self.length = len(self.keys)
        self.mixture_dataset = mixture_dataset_dict
        self.clean_dataset = clean_dataset_dict

        print(f"Finish, len(finial dataset) == {self.length}.")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        name = self.keys[item]
        num = name.split("_")[0]
        mixture = self.mixture_dataset[name]
        clean = self.clean_dataset[num]

        assert mixture.shape == clean.shape
        return mixture.reshape(-1), clean.reshape(-1), name
