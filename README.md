# IRM based Speech Enhancement using DNN

Inspired by funcwj's [nn-ideal-mask](https://github.com/funcwj/nn-ideal-mask). Compared to his project:

- Increased visualization of validation data
- More structured
- Support PyTorch v1.1


## ToDo

- [ ] 更新至 PyTorch 1.1，使用 PyTorch 内置的 Tensorboard 相关函数替换掉 TensorboardX
- [ ] 新增多帧到单帧映射的数据集
- [ ] 构建适合现有数据集的 DataLoader，并整合至 train.py 中
- [ ] 参考 SNR-based DL DNN 构建验证逻辑，以及可视化
- [ ] 跑通模型