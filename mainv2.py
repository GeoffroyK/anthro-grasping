import datetime
import os
import sys
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from traning import train, validate
from utils.data import get_dataset
from models.swinv2 import SwinTransformerV2

import tensorboardX

import numpy as np
from GPUtil import showUtilization as gpu_usage

gpu_usage()

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="TF-Grasp")

    # Network
    # Dataset & Data & Training
    parser.add_argument(
        "--dataset",
        type=str,
        default="cornell",
        help='Dataset Name ("cornell" or "jacquard")',
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/media/pollen/T7/cornell",
        help="Path to dataset",
    )
    parser.add_argument(
        "--use-depth", type=int, default=1, help="Use Depth image for training (1/0)"
    )
    parser.add_argument(
        "--use-rgb", type=int, default=1, help="Use RGB image for training (0/1)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Fraction of data for training (remainder is validation)",
    )
    parser.add_argument(
        "--ds-rotate",
        type=float,
        default=0.0,
        help="Shift the start point of the dataset to use a different test/train split for cross validation.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Dataset workers")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--vis", type=bool, default=False, help="vis")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument(
        "--batches-per-epoch", type=int, default=200, help="Batches per Epoch"
    )
    parser.add_argument(
        "--val-batches", type=int, default=32, help="Validation Batches"
    )
    # Logging etc.
    parser.add_argument(
        "--description", type=str, default="", help="Training description"
    )
    parser.add_argument(
        "--outdir", type=str, default="output/models/", help="Training Output Directory"
    )
    parser.add_argument(
        "--snapshot", type=str, default=None, help="Snapshot save directory"
    )
    parser.add_argument(
        "--logdir", type=str, default="tensorboard/", help="Log directory"
    )
    args = parser.parse_args()
    return args


global_mean_loss = []


def run():
    args = parse_args()
    dt = datetime.datetime.now().strftime("%y%m%d_%H%M")
    net_desc = "{}_{}".format(dt, "_".join(args.description.split()))
    save_folder = os.path.join(args.outdir, net_desc)
    writer = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info("Loading {} Dataset...".format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(
        args.dataset_path,
        start=0.0,
        end=args.split,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
    )
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = Dataset(
        args.dataset_path,
        start=args.split,
        end=1.0,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    logging.info("Done")

    logging.info("Loading Network...")
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    net = SwinTransformerV2(
        in_chans=input_channels, embed_dim=48, num_heads=[1, 2, 4, 8]
    )
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    listy = [x * 2 for x in range(1, 1000, 5)]
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=listy, gamma=0.5
    )
    logging.info("Done")
    # summary(net, (input_channels, 224, 224))
    f = open(os.path.join(save_folder, "net.txt"), "w")
    sys.stdout = f
    # summary(net, (input_channels, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()
    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info("Beginning Epoch {:02d}".format(epoch))
        print("current lr:", optimizer.state_dict()["param_groups"][0]["lr"])
        # for i in range(5000):
        train_results, mean_loss = train(
            epoch,
            net,
            device,
            train_data,
            optimizer,
            args.batches_per_epoch,
            vis=args.vis,
        )

        print(f"Global mean loss for this epoch {global_mean_loss}")

        # schedule.step()
        # Run Validation[
        logging.info("Validating...")
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info(
            "%d/%d = %f"
            % (
                test_results["correct"],
                test_results["correct"] + test_results["failed"],
                test_results["correct"]
                / (test_results["correct"] + test_results["failed"]),
            )
        )

        iou = test_results["correct"] / (
            test_results["correct"] + test_results["failed"]
        )

        writer.add_scalar("Mean Loss / Epoch", mean_loss, epoch)
        writer.add_scalar("loss", test_results["loss"], epoch)

        writer.add_scalar("loss/val_loss", test_results["loss"], epoch)
        for n, l in test_results["losses"].items():
            writer.add_scalar("val_loss/" + n, l, epoch)

        if epoch % 1 == 0 or iou > best_iou:
            torch.save(
                net, os.path.join(save_folder, "epoch_%02d_iou_%0.2f" % (epoch, iou))
            )
            torch.save(
                net.state_dict(),
                os.path.join(
                    save_folder, "epoch_%02d_iou_%0.2f_statedict.pt" % (epoch, iou)
                ),
            )

            if epoch % 50 == 0:
                # Checkpoint all 50 epochs
                state = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state, os.path.join(save_folder, "snapshot_epoch%02d.pt" % (epoch))
                )
        best_iou = iou


if __name__ == "__main__":
    run()
