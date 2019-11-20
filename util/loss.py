import torch
from torch.nn.utils.rnn import pad_sequence


def mse_loss():
    return torch.nn.MSELoss()


def l1_loss():
    return torch.nn.L1Loss()


def bce_loss():
    return torch.nn.BCEWithLogitsLoss()


def mse_loss_for_variable_length_data():
    def loss_function(ipt, target, n_frames_list):
        """Calculate the MSE loss for variable length dataset.
        """
        E = 1e-7
        with torch.no_grad():
            masks = []
            for n_frames in n_frames_list:
                masks.append(torch.ones((n_frames, target.size(2)), dtype=torch.float32))  # the shape is (T, F)

            binary_mask = pad_sequence(masks, batch_first=True).cuda()

        masked_ipt = ipt * binary_mask
        masked_target = target * binary_mask
        return ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)

    return loss_function
