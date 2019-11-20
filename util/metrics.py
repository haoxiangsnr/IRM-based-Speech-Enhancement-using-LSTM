from pypesq import pesq
from pystoi.stoi import stoi


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def _compute_PESQ_sub_task(clean_signal, noisy_siganl, sr=16000):
    return pesq(clean_signal, noisy_siganl, sr)


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(clean_signal, noisy_signal, sr)