import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.train_dataset import TrainDataset
from trainer.trainer import Trainer
from utils.utils import initialize_config


def main(config, resume):
    """训练脚本的入口函数

    Steps:
        1. 加载数据集
        2. 初始化模型
        3. 设置优化器
        4. 选择损失函数
        5. 训练脚本 run

    Arguments:
        config {dict} -- 配置项
        resume {bool} -- 是否加载最近一次存储的模型断点
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_dataset = TrainDataset(
        mixture_dataset=config["train_dataset"]["mixture"],
        clean_dataset=config["train_dataset"]["clean"],
        limit=config["train_dataset"]["limit"],
        offset=config["train_dataset"]["offset"],
        apply_normalization=config["train_dataset"]["apply_normalization"]
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_dataset"]["batch_size"],
        num_workers=config["train_dataset"]["num_workers"],
        shuffle=config["train_dataset"]["shuffle"]
    )

    valid_dataset = TrainDataset(
        mixture_dataset=config["valid_dataset"]["mixture"],
        clean_dataset=config["valid_dataset"]["clean"],
        limit=config["valid_dataset"]["limit"],
        offset=config["valid_dataset"]["offset"],
        apply_normalization=config["valid_dataset"]["apply_normalization"]
    )

    valid_data_loader = DataLoader(
        dataset=valid_dataset
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"]
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optim=optimizer,
        train_dl=train_data_loader,
        validation_dl=valid_data_loader,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='IRM Estimation using DNN in Speech Enhancement')
    parser.add_argument("-C", "--config", required=True,
                        type=str, help="训练配置文件（*.json）")
    parser.add_argument('-D', '--device', default=None,
                        type=str, help="本次实验使用的 GPU 索引，e.g. '1,2,3'")
    parser.add_argument("-R", "--resume", action="store_true",
                        help="是否从最近的一个断点处继续训练")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # load config file
    config = json.load(open(args.config))
    config["train_config_path"] = args.config

    main(config, resume=args.resume)
