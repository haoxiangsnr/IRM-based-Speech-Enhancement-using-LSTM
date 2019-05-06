from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.utils import sample_fixed_length_data_aligned, apply_mean_std

class TrainDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self, mixture_dataset, clean_dataset, limit=None, offset=0):
        """
        构建训练数据集

        Args:
            mixture_dataset (str): 带噪语音数据集
            clean_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
            apply_normalization (bool): 是否对输入进行规范化（减均值，除标准差）
        """
        self.apply_normalization = apply_normalization
        mixture_dataset = Path(mixture_dataset)
        clean_dataset = Path(clean_dataset)

        assert mixture_dataset.exists() and clean_dataset.exists(), "训练数据集不存在"

        print(f"Loading mixture dataset {mixture_dataset.as_posix()} ...")
        self.mixture_dataset:dict = np.load(mixture_dataset.as_posix()).item()
        print(f"Loading clean dataset {clean_dataset.as_posix()} ...")
        self.clean_dataset:dict = np.load(clean_dataset.as_posix()).item()
        assert len(self.mixture_dataset) % len(self.clean_dataset) == 0, \
            "mixture dataset 的长度不是 clean dataset 的整数倍"

        print(f"The len of fully dataset is {len(self.mixture_dataset)}.")
        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")

        self.mapping_table = {}
        self.length = 0
        for k, v in self.mixture_dataset.items():
            assert v.shape[1] % 7 == 0
            self.mapping_table[k] = v.shape[1] // 7
            self.length += v.shape[1]

        # if limit is None:
        #     limit = len(self.mixture_dataset)
        #
        # self.keys = list(self.mixture_dataset.keys()) # 语句数量
        # self.keys.sort()
        # self.keys = self.keys[offset: offset + limit]
        #
        # self.length = len(self.keys)
        print(f"Finish, len(finial dataset) == {self.length}.")

    def __len__(self):
        return self.length

    def __getitem__(self, frame_index):
        i, n_pad = 0, 7

        mixture_frames = None
        clean_frame = None
        name = ""

        # e.g. ("0001_babble_-5", 238)
        mapping_table_keys = list(self.mapping_table.keys())
        for i_in_dict, (name, n_frames) in enumerate(self.mapping_table.items()):
            if i + n_frames < frame_index:
                i += n_frames
            elif i + n_frames > frame_index:
                mixture_lps = self.mixture_dataset[name]
                clean_lps = self.clean_dataset[name.split("_")[0]]

                mixture_offset_in_lps = (frame_index - i) * n_pad # 当前 7 帧在 lps 中偏移值
                clean_offset_in_lps = frame_index - i # 当前 1 帧在 lps 中偏移值

                mixture_frames = mixture_lps[:, mixture_offset_in_lps: (mixture_offset_in_lps + n_pad)]
                clean_frame = clean_lps[:, clean_offset_in_lps].reshape(-1, 1)
                break
            else:
                name = mapping_table_keys[i_in_dict + 1]
                mixture_lps = self.mixture_dataset[name] # 最开始的 7 帧
                clean_lps = self.clean_dataset[name.split("_")[0]] # 首帧
                mixture_frames = mixture_lps[:, :n_pad]
                clean_frame = clean_lps[:, 0].reshape(-1, 1)
                break

        assert mixture_frames.shape == (257, 7) and clean_frame.shape == (257, 1)
        return mixture_frames, clean_frame, name
