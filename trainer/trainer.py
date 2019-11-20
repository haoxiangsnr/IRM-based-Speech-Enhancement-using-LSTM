import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from util.metrics import compute_PESQ, compute_STOI

plt.switch_backend("agg")


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            optimizer,
            loss_function,
            train_dl,
            validation_dl,
    ):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy_mag, _, mask in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy_mag = noisy_mag.to(self.device)

            pred_mask = self.model(noisy_mag)
            loss = self.loss_function(pred_mask, mask)
            loss_total += loss

            loss.backward()
            self.optimizer.step()

        self.writer.add_scalar("Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        for i, (noisy_y, clean_y, name) in enumerate(self.validation_dataloader):
            assert len(name) == 1, "The batch size of validation dataloader must be 1."
            name = name[0]

            noisy_y = noisy_y.numpy().reshape(-1)
            clean_y = clean_y.numpy().reshape(-1)

            noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy_y, n_fft=320, hop_length=160, win_length=320))

            noisy_mag_tensor = torch.tensor(noisy_mag[None, ...], device=self.device).permute(0, 2, 1)  # (batch_size, T, F)
            assert noisy_mag_tensor.dim() == 3

            pred_mask = self.model(noisy_mag_tensor)
            pred_clean_mag_tensor = noisy_mag_tensor * pred_mask

            pred_clean_mag = torch.t(pred_clean_mag_tensor.squeeze(0)).detach().cpu().numpy()  # (F, T)
            pred_clean_y = librosa.istft(pred_clean_mag * noisy_phase, hop_length=160, win_length=320)

            assert len(clean_y) == len(pred_clean_y) == len(noisy_y)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", noisy_y, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Clean", pred_clean_y, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Denoisy", clean_y, epoch, sample_rate=16000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([noisy_y, pred_clean_y, clean_y]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            clean_mag, _ = librosa.magphase(librosa.stft(clean_y, n_fft=320, hop_length=160, win_length=320))
            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    pred_clean_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k])
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metrics
            stoi_c_n.append(compute_STOI(clean_y, noisy_y, sr=16000))
            stoi_c_d.append(compute_STOI(clean_y, pred_clean_y, sr=16000))
            pesq_c_n.append(compute_PESQ(clean_y, noisy_y, sr=16000))
            pesq_c_d.append(compute_PESQ(clean_y, pred_clean_y, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metrics/STOI", {
            "clean and noisy": get_metrics_ave(stoi_c_n),
            "clean and denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.writer.add_scalars(f"Metrics/PESQ", {
            "clean and noisy": get_metrics_ave(pesq_c_n),
            "clean and denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        return score
