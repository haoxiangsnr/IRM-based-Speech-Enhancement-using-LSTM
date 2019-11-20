import argparse
import os

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from util.utils import initialize_config, pad_to_longest_in_one_batch


def main(config, resume):
    torch.manual_seed(config["seed"])  # both CPU and GPU
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"],  # Set it to False for very small dataset, otherwise True.
        collate_fn=pad_to_longest_in_one_batch
    )
    validation_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        batch_size=1,
        num_workers=1,
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_dl=train_dataloader,
        validation_dl=validation_dataloader,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRM based speech enhancement using LSTM")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
