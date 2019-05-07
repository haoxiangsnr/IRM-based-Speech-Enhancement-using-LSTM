from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self,
                 mixture_dataset="/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/DNN/pad_3_400_50/train/mixture.npy",
                 mask_dataset="/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/DNN/pad_3_400_50/train/mask.npy",
                 limit=None,
                 offset=0):
        """
        构建训练数据集

        Args:
            mixture_dataset (str): 带噪语音数据集
            mask_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        mixture_dataset = Path(mixture_dataset)
        mask_dataset = Path(mask_dataset)

        assert mixture_dataset.exists() and mask_dataset.exists(), "训练数据集不存在"

        print(f"Loading mixture dataset {mixture_dataset.as_posix()} ...")
        mixture_dataset_dict: dict = np.load(mixture_dataset.as_posix()).item()

        print(f"Loading mask dataset {mask_dataset.as_posix()} ...")
        mask_dataset_dict: dict = np.load(mask_dataset.as_posix()).item()

        # 我的内存足够大，内存速度也足够快，我以空间换时间
        mixture_dataset = []
        mask_dataset = []
        for mixture_name, mixture_padded_lps in mixture_dataset_dict.items():
            # [ [257,7], [257, 7], ...]
            mixture_lps_chunks = np.split(mixture_padded_lps, mixture_padded_lps.shape[1] // 7, axis=1)
            mixture_dataset += mixture_lps_chunks

            # [[257, 1], [257, 1]]
            mask_lps = mask_dataset_dict[mixture_name]
            mask_lps_chunks = np.split(mask_lps, mask_lps.shape[1], axis=1)
            mask_dataset += mask_lps_chunks

        assert len(mixture_dataset) == len(mask_dataset)

        self.length = len(mixture_dataset)
        self.mixture_dataset = mixture_dataset
        self.mask_dataset = mask_dataset

        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")
        print(f"Finish, len(finial dataset) == {self.length}.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mixture = self.mixture_dataset[idx].reshape(-1) # DNN 作为网络，要求铺平
        mask = self.mask_dataset[idx].reshape(-1)
        return mixture, mask
