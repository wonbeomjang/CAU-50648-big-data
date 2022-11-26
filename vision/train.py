import argparse
from typing import Dict, Optional

import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision.dataloader import get_loader
from vision.logger import Logger
from vision.models import PogModel
from utils import AverageMeter


def train_one_epoch(net: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                    lr_scheduler: Optional[optim.lr_scheduler.OneCycleLR], epoch: int, total_epochs: int) -> Dict[str, float]:
    device = next(net.parameters()).device
    net = net.train()
    loss_avg = AverageMeter()

    pbar = tqdm(dataloader)
    for images, targets in pbar:
        lr = optimizer.param_groups[0]["lr"]
        images: Tensor = images.to(device)
        targets: Tensor = targets.to(device)

        preds = net(images)
        loss: Tensor = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_avg.update(float(loss.data))
        pbar.set_description(f"[{epoch}/{total_epochs}] Lr: {lr} Loss: {loss_avg.avg:.4f}")

    return {"train_loss": loss_avg.avg}


def val(net: nn.Module, dataloader: DataLoader, criterion: nn.Module, epoch: int,
        total_epochs: int) -> Dict[str, float]:

    with torch.no_grad():
        device = next(net.parameters()).device
        net = net.eval()
        loss_avg = AverageMeter()

        pbar = tqdm(dataloader)
        for images, targets in pbar:
            images: Tensor = images.to(device)
            targets: Tensor = targets.to(device)

            preds = net(images)
            loss: Tensor = criterion(preds, targets)

            loss_avg.update(float(loss.data))
            pbar.set_description(f"[{epoch}/{total_epochs}] Validation... Loss: {loss_avg.avg:.4f}")

    return {"val_loss": loss_avg.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--competition_name", type=str, default="kaggle-pog-series-s01e01")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = PogModel().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), args.lr)
    train_loader, val_loader = get_loader(args.batch_size, image_size=args.image_size,
                                          competition_name=args.competition_name)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    start_epoch = 0

    logger = Logger("bigdata", args, resume=args.resume)
    if args.resume:
        state_dict = logger.load_state_dict(map_location=device)
        net.load_state_dict(state_dict["state_dict"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        start_epoch = state_dict["epoch"]
        logger.metrix = state_dict["metrix"]

    for i in range(start_epoch, args.num_epochs):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, None, i, args.num_epochs)
        val_loss = val(net, val_loader, criterion, i, args.num_epochs)
        lr_scheduler.step(val_loss["val_loss"])

        logger.log(train_loss)
        logger.log(val_loss)
        logger.log({"lr": optimizer.param_groups[0]["lr"]})
        logger.save_model(net, optimizer, lr_scheduler, metrix=val_loss["val_loss"], epoch=i)
        logger.end_epoch()

    logger.finish()
    