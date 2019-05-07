import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from utils.metrics import compute_PESQ, compute_STOI
from utils.utils import ExecutionTime, unfold_spectrum, phase, rebuild_waveform, mag, \
    input_normalization


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optim,
            train_dl,
            validation_dl,
    ):
        super(Trainer, self).__init__(
            config, resume, model, loss_function, optim)
        self.train_data_loader = train_dl
        self.validation_data_loader = validation_dl

    def _set_model_train(self):
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/13
        self.model.train()

    def _set_model_eval(self):
        self.model.eval()

    def _train_epoch(self, epoch):
        """
        定义单次训练的逻辑

        Args:
            epoch: 当前实验到了哪个轮次

        Steps:
            1. 设置模型运行状态
            2. 从 DataLoader 中获取 input 和 target
            3. 计算损失
            4. 反向传播梯度，更新参数
            5. 可视化损失
        """
        self._set_model_train()
        loss_total = 0.0
        for i, (data, target) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()

            data = data.to(self.dev)
            target = target.to(self.dev)

            output = self.model(data)
            loss = self.loss_function(output, target)
            loss_total += loss

            loss.backward()
            self.optimizer.step()

        visualize_loss = lambda tag, value: self.viz.writer.add_scalar(f"训练损失/{tag}", value, epoch)
        # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/4
        # The length of the loader will adapt to the batch_size
        visualize_loss("loss", loss_total / len(self.train_data_loader))

    def _validation_epoch(self, epoch):
        """
        定义验证逻辑

        Notes:
            验证时使用测试的 DataLoader，并且 batch_size 与 num_workers 均为 1

        Args:
            epoch: 当前实验进行到了哪个轮次

        Steps:
            1. 从 DataLoader 中取 mixture 语音
            2. 计算 mixture 的线性频谱并进行 overlap 拓展分组，每 7 帧 mixture 为一组
            3. 遍历 mixture 的分组结果，每 7 帧 mixture 送入模型得到 1 帧预测的 mask，拼接所有的 mask 帧
            4. 通过 mask 计算出 enhanced 线性频谱
            5. 将 enhanced 还原为波形文件，并计算评价指标
        """
        self._set_model_eval()
        stoi_clean_mixture = []
        stoi_clean_enhanced = []
        pesq_clean_mixture = []
        pesq_clean_enhanced = []

        with torch.no_grad():
            for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
                mixture = mixture[0].numpy()
                clean = clean[0].numpy()
                name = name[0]

                # 注意与数据集中保持一致
                mixture_mag = input_normalization(mag(mixture))
                mixture_phase = phase(mixture)

                mixture_mag_padded = torch.Tensor(unfold_spectrum(mixture_mag, n_pad=3))

                # mixture_frames => mask_frame
                mask_mag = []
                for mixture_frames in torch.chunk(mixture_mag_padded, mixture_mag_padded.shape[1] // 7, dim=1):
                    mixture_frames = mixture_frames.reshape(1, -1) # (batch_size, 257 * 7)
                    assert mixture_frames.shape == (1, 257 * 7)
                    mixture_frames = mixture_frames.to(self.dev)

                    mask_frame = self.model(mixture_frames).cpu().numpy().reshape(-1, 1)
                    assert mask_frame.shape == (257, 1)
                    mask_mag.append(mask_frame)

                mask_mag = np.concatenate(mask_mag, axis=1)
                assert mixture_mag_padded.shape[1] / 7 == mask_mag.shape[1], "每 7 帧 mixture 对应 1 帧 mask"

                # 重建语音波形
                enhanced_mag = mixture_mag * (1 - mask_mag)
                enhanced = rebuild_waveform(enhanced_mag, mixture_phase)

                # 修复长度误差
                min_length = min(len(mixture), len(enhanced), len(clean))
                mixture = mixture[:min_length]
                enhanced = enhanced[:min_length]
                clean = clean[:min_length]

                self.viz.writer.add_audio(f"语音文件/{name[0]}带噪语音", mixture, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"语音文件/{name[0]}降噪语音", enhanced, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"语音文件/{name[0]}纯净语音", clean, epoch, sample_rate=16000)

                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()

                self.viz.writer.add_figure(f"语音波形图像/{name}", fig, epoch)

                stoi_clean_mixture.append(compute_STOI(clean, mixture, sr=16000))
                stoi_clean_enhanced.append(compute_STOI(clean, enhanced, sr=16000))
                pesq_clean_mixture.append(compute_PESQ(clean, mixture, sr=16000))
                pesq_clean_enhanced.append(compute_PESQ(clean, enhanced, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"评价指标均值/STOI", {
            "clean 与 noisy": get_metrics_ave(stoi_clean_mixture),
            "clean 与 denoisy": get_metrics_ave(stoi_clean_enhanced)
        }, epoch)
        self.viz.writer.add_scalars(f"评价指标均值/PESQ", {
            "clean 与 noisy": get_metrics_ave(pesq_clean_mixture),
            "clean 与 denoisy": get_metrics_ave(pesq_clean_enhanced)
        }, epoch)

        score = (get_metrics_ave(stoi_clean_enhanced) +
                 self._transform_pesq_range(get_metrics_ave(pesq_clean_enhanced))) / 2
        return score

    def _transform_pesq_range(self, pesq_score):
        """平移 PESQ 评价指标
        将 PESQ 评价指标的范围从 -0.5 ~ 4.5 平移为 0 ~ 1

        Args:
            pesq_score: PESQ 得分

        Returns:
            0 ~ 1 范围的 PESQ 得分

        """
        return (pesq_score + 0.5) * 2 / 10

    def _is_best_score(self, score):
        """检查当前的结果是否为最佳模型"""
        if score >= self.best_score:
            self.best_score = score
            return True
        return False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============ Train epoch = {epoch} ============")
            print("[0 seconds] 开始训练...")
            timer = ExecutionTime()
            self.viz.set_epoch(epoch)

            self._train_epoch(epoch)

            if self.visualize_metrics_period != 0 and epoch % self.visualize_metrics_period == 0:
                # 测试一轮，并绘制波形文件
                print(f"[{timer.duration()} seconds] 训练结束，开始计算评价指标...")
                score = self._validation_epoch(epoch)

                if self._is_best_score(score):
                    self._save_checkpoint(epoch, is_best=True)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            print(f"[{timer.duration()} seconds] 完成当前 Epoch.")
