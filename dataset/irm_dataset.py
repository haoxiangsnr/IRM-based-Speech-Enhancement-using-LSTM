import os
import random

import librosa
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from util.utils import synthesis_noisy_y


class IRMDataset(Dataset):
    def __init__(self,
                 noise_dataset="/home/imucs/Datasets/Build-SE-Dataset-V2/Data/noise.txt",
                 clean_dataset="/home/imucs/Datasets/Build-SE-Dataset-V2/Data/clean.txt",
                 snr_list=None,
                 offset=700,
                 limit=None,
                 mode="train",
                 n_jobs=-1
                 ):
        """Construct training dataset.

        Args:
            noise_dataset: List, which saved the paths of noise files.
            clean_dataset: List, which saved the paths of clean wav files.
            offset: offset of clean_dataset.
            limit: limit of clean_dataset from offset position.
            n_jobs: Use multithreading to pre-load noise files, see joblib (https://joblib.readthedocs.io/en/latest/parallel.html).
            mode:
                "train": return noisy magnitude, mask.
                "validation": return noisy_y, clean_y, name
                "test": return noisy_y, name

        Notes:
            clean_dataset (*.txt):
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FSJK1/SI696.WAV
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FSJK1/SI696.WAV
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FSJK1/SI696.WAV
                ...

            noise_dataset (*.txt):
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/Noise/bus.wav
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/Noise/cafe.wav
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/Noise/ped.wav
                /home/imucs/Datasets/Build-SE-Dataset-V2/Data/Noise/str.wav
                ...
        """
        super().__init__()
        assert mode in ["train", "validation", "test"], "mode parameter must be one of 'train', 'validation', and 'test'."
        clean_f_paths = [line.rstrip('\n') for line in open(clean_dataset, "r")]
        clean_f_paths = clean_f_paths[offset:]
        if limit:
            clean_f_paths = clean_f_paths[:limit]

        noise_f_paths = [line.rstrip('\n') for line in open(noise_dataset, "r")]

        def load_noise_file(file_path, sr=16000):
            basename_text = os.path.basename(os.path.splitext(file_path)[0])
            y, _ = librosa.load(file_path, sr=sr)
            return {
                "name": basename_text,
                "y": y
            }

        # At the initialization of the model, use multithreading to load noise.
        # "-1" means use all CPUs.
        # "1" means not apply multithreading.
        # "3" means use 3 CPUs.
        all_noise_data = Parallel(n_jobs=n_jobs)(delayed(load_noise_file)(f_path, sr=16000) for f_path in tqdm(noise_f_paths, desc=f"Loading {mode} noise files"))

        self.length = len(clean_f_paths)
        self.all_noise_data = all_noise_data
        self.clean_f_paths = clean_f_paths
        self.snr_list = snr_list
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean_y, _ = librosa.load(self.clean_f_paths[idx], sr=16000)
        snr = random.choice(self.snr_list)

        noise_data = random.choice(self.all_noise_data)
        noise_name = noise_data["name"]
        noise_y = noise_data["y"]

        name = f"{str(idx).zfill(5)}_{noise_name}_{snr}"
        clean_y, noise_y, noisy_y = synthesis_noisy_y(clean_y, noise_y, snr)

        if self.mode == "train":
            clean_mag, _ = librosa.magphase(librosa.stft(clean_y, n_fft=320, hop_length=160, win_length=320))
            noise_mag, _ = librosa.magphase(librosa.stft(noise_y, n_fft=320, hop_length=160, win_length=320))
            noisy_mag, _ = librosa.magphase(librosa.stft(noisy_y, n_fft=320, hop_length=160, win_length=320))
            mask = np.sqrt(clean_mag ** 2 / (clean_mag + noise_mag) ** 2)
            n_frames = clean_mag.shape[-1]
            return noisy_mag, clean_mag, mask, n_frames
        elif self.mode == "validation":
            return noisy_y, clean_y, name
        else:
            return noisy_y, name


if __name__ == '__main__':
    dataset = IRMDataset(snr_list=["-5", "10"])
    res = next(iter(dataset))
    print(res[0].shape)
    print(res[1].shape)
    print(res[2])
