import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from utils.metrics import compute_PESQ, compute_STOI
from utils.utils import ExecutionTime, cal_lps, unfold_spectrum, phase, lps_to_mag, rebuild_waveform


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
        """定义单次训练的逻辑
        
        Steps:
            1. 设置模型运行状态
            2. 从 DataLoader 中获取 input 和 target
            3. 计算损失
            4. 反向传播梯度，更新参数
            5. 可视化损失

        Arguments:
            epoch {int} -- 当前实验到了哪个轮次
        """
        self._set_model_train()
        loss_total = 0.0
        for i, (data, target, basename_text) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()

            data = data.to(self.dev)
            target = target.to(self.dev)

            output = self.model(data)
            loss = self.loss_function(output, target)
            loss_total += loss

            loss.backward()
            self.optimizer.step()

        # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/4
        # The length of the loader will adapt to the batch_size
        dl_len = len(self.train_data_loader)

        def visualize_loss(tag, total): return self.viz.writer.add_scalar(
            f"训练损失/{tag}", total / dl_len, epoch)
        visualize_loss("loss", loss_total)

    def _validation_epoch(self, epoch):
        """
        验证轮，验证时使用验证集，且 batch_size 与 num_workers 均为 1。
        """

        self._set_model_eval()
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        with torch.no_grad():
            for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
                """
                Notes:
                    1. 提取带噪语音 LPS 特征与原始相位
                    2. 将带噪语音 LPS 特征做拓展，并分帧
                    3. 每 7 帧带噪 LPS 特征送入网络中，得到一帧降噪过的语音
                    4. 拼接降噪过的语音，使用带噪语音的相位还原为时域信号
                """
                clean = clean[0]
                name = name[0]
                mixture = mixture[0]

                mixture_lps = cal_lps(mixture)
                mixture_phase = phase(mixture)

                mixture_lps = unfold_spectrum(mixture_lps, n_pad=3)

                enhanced_lps = []
                for mixture_frames in torch.chunk(mixture_lps, mixture_lps.shape[1] // 7, dim=1):
                    enhanced_frame = self.model(mixture_frames.to(self.dev)).reshape(-1, 1).cpu().numpy()
                    enhanced_lps.append(enhanced_frame)

                enhanced_lps = np.concatenate(enhanced_lps, axis=1)
                assert mixture_frames.shape == (257, 7)
                assert mixture_lps.shape[1] / 7 == enhanced_lps.shape[1]

                enhanced_mag = lps_to_mag(enhanced_lps)
                enhanced = rebuild_waveform(enhanced_mag, mixture_phase)

                min_length = min(len(mixture), len(enhanced), len(clean))
                mixture = mixture[:min_length]
                enhanced = enhanced[:min_length]
                clean = clean[:min_length]

                self.viz.writer.add_audio(
                    f"语音文件/{name[0]}带噪语音", mixture, epoch, sample_rate=16000)
                self.viz.writer.add_audio(
                    f"语音文件/{name[0]}降噪语音", enhanced, epoch, sample_rate=16000)
                self.viz.writer.add_audio(
                    f"语音文件/{name[0]}纯净语音", clean, epoch, sample_rate=16000)

                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        torch.mean(y),
                        torch.std(y),
                        torch.max(y),
                        torch.min(y)
                    ))
                    librosa.display.waveplot(
                        y.cpu().squeeze().numpy(), sr=16000, ax=ax[j])
                plt.tight_layout()

                self.viz.writer.add_figure(
                    f"语音波形图像/{name}", fig, epoch)

                ny = mixture.cpu().numpy().reshape(-1)
                dy = enhanced.cpu().numpy().reshape(-1)
                cy = clean.cpu().numpy().reshape(-1)

                stoi_c_n.append(compute_STOI(cy, ny, sr=16000))
                stoi_c_d.append(compute_STOI(cy, dy, sr=16000))
                pesq_c_n.append(compute_PESQ(cy, ny, sr=16000))
                pesq_c_d.append(compute_PESQ(cy, dy, sr=16000))

        def get_metrics_ave(metrics): return np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"评价指标均值/STOI", {
            "clean 与 noisy": get_metrics_ave(stoi_c_n),
            "clean 与 denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.viz.writer.add_scalars(f"评价指标均值/PESQ", {
            "clean 与 noisy": get_metrics_ave(pesq_c_n),
            "clean 与 denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        score = (get_metrics_ave(stoi_c_d) +
                 self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
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
