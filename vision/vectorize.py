import argparse

import pandas as pd
import psutil
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import PogDataset
from models import PogModel
from utils import AverageMeter


def get_loader(batch_size, competition_name: str = "kaggle-pog-series-s01e01", image_size: int = 256, scale: int = 1):
    dataset = PogDataset(competition_name, image_size, scale)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader


def vectorize(net: nn.Module, dataloader: DataLoader):
    with torch.no_grad():
        device = next(net.parameters()).device
        net = net.eval()
        loss_avg = AverageMeter()

        result = []

        pbar = tqdm(dataloader)
        for images, targets, image_id in pbar:
            images: Tensor = images.to(device)

            preds = net(images)

            res = pd.DataFrame(preds.cpu())
            res["image_id"] = image_id
            result += [res]

    return pd.concat(result, axis=0, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--competition_name", type=str, default="kaggle-pog-series-s01e01")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = PogModel().backbone.to(device)
    data_loader = get_loader(args.batch_size, image_size=args.image_size,
                             competition_name=args.competition_name)

    # net.load_state_dict(state_dict["state_dict"])

    result = vectorize(net, data_loader)

    print(result)
