import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from typing import Optional
from fastai.vision.all import *
import numpy as np
import torch
import torch.nn as nn


def conv_block(ni, nf, size=3, stride=1):

    adjust_kernel = lambda s: s if s > 2 else 3
    return nn.Sequential(
        nn.Conv2d(
            ni, nf,
            kernel_size=size,
            stride=stride,
            padding=(adjust_kernel(size) - 1) // 2,
            bias=False
        ),
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


def triple_conv(ni, nf):

    return nn.Sequential(
        conv_block(ni, nf),
        conv_block(nf, ni, size=1),
        conv_block(ni, nf),
    )


def maxpooling():
    return nn.MaxPool2d(2, stride=2)


# def build_darkcovidnet(num_classes: int = 3) -> nn.Module:

#     model = nn.Sequential(
#         conv_block(3, 16),       # Output: 256x256x16 
#         maxpooling(),            # 128x128x16
#         triple_conv(16, 32),     # 128x128x32
#         maxpooling(),            # 64x64x32
#         triple_conv(32, 64),     # 64x64x64
#         maxpooling(),            # 32x32x64
#         conv_block(64, 128),     # 32x32x128
#         maxpooling(),            # 16x16x128
#         nn.Flatten(),            # 16*16*128 = 32768
#         nn.Linear(16 * 16 * 128, num_classes)
#         # nn.AdaptiveAvgPool2d(1),
#         # nn.Flatten(),
#         # nn.Dropout(0.3),
#         # nn.Linear(128, num_classes)
#     )
#     return model

def build_darkcovidnet(num_classes: int = 3) -> nn.Module:
    model = nn.Sequential(
        conv_block(3, 16),
        maxpooling(),
        triple_conv(16, 32),
        maxpooling(),
        triple_conv(32, 64),
        maxpooling(),
        conv_block(64, 128),
        maxpooling(),              # (B, 128, 16, 16)
        nn.AdaptiveAvgPool2d(1),   # (B, 128, 1, 1)
        nn.Flatten(),              # (B, 128)
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

def get_dataloaders(
    data_dir: Path,
    bs: int = 32,
    num_workers: int = 4
) -> ImageDataLoaders:

    dls = ImageDataLoaders.from_folder(
        path=data_dir,
        train='train',
        valid='valid',
        shuffle=True,
        batch_tfms=[Normalize.from_stats(*imagenet_stats)],
        bs=bs,
        num_workers=num_workers
    )
    return dls



def train_model(
    dls: ImageDataLoaders,
    epochs: int = 60,
    lr: Optional[float] = None
) -> Learner:

    num_classes = len(dls.vocab)
    print(f"Classes: {dls.vocab}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(dls.train_ds)}")
    print(f"Number of validation samples: {len(dls.valid_ds)}")

    model = build_darkcovidnet(num_classes=num_classes)

    learn = Learner(
        dls,
        model,
        loss_func=nn.CrossEntropyLoss(),
        metrics=accuracy,
        opt_func=RAdam
    )

    print(learn.summary())

    if lr is not None:
        learn.fit_one_cycle(epochs, lr, wd=3e-3)
    else:
        learn.fit_one_cycle(epochs, wd=3e-3)

    return learn


def evaluate_model(learn: Learner):

    dls = learn.dls
    print("Number of validation examples:", len(dls.valid_ds))

    probs, targets = learn.get_preds(dl=dls.valid)
    acc = accuracy(probs, targets)

    # Cast to float or .item()
    acc_float = float(acc)
    print(f"fastai accuracy: {acc_float:.4f}")

    preds_np = probs.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    correct = (preds_np == targets_np).sum()
    total = len(preds_np)
    acc_manual = correct / total
    print(f"Manual accuracy: {correct}/{total} = {acc_manual:.4f}")

    np.set_printoptions(threshold=np.inf)
    cm = confusion_matrix(targets_np, preds_np)
    print("Confusion matrix:")
    print(cm)

    target_names = [str(v) for v in dls.vocab]
    print("Classification report:")
    print(classification_report(targets_np, preds_np, target_names=target_names))


def parse_args():
    parser = argparse.ArgumentParser(
        description="DarkCovidNet 3-class training script (converted from notebook)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="thyroid_dataset_more_classes_full_3class",
        help=(
            "Path to the root dataset folder. "
            "Inside it must contain 'train' and 'valid' subfolders. "
            "Example: /path/to/thyroid_dataset_more_classes_full_3class"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60).",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for fit_one_cycle. If None, fastai will choose automatically.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help=(
            "Optional: path to save the FastAI trained model as a .pkl file. "
            "Example: --save-model darkcovidnet_3class.pkl"
        ),
    )
    
    parser.add_argument(
    "--save-torch",
    type=str,
    default=None,
    help=(
        "Optional: path to save raw PyTorch state_dict. "
        "Example: --save-torch darkcovidnet_3class_state.pth"
    ),
)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data_dir {data_dir} does not exist. "
            f"Please check the --data-dir argument."
        )

    print(f"Using data directory: {data_dir.resolve()}")

    dls = get_dataloaders(
        data_dir=data_dir,
        bs=args.bs,
        num_workers=args.num_workers,
    )

    learn = train_model(
        dls=dls,
        epochs=args.epochs,
        lr=args.lr,
    )

    evaluate_model(learn)

    if args.save_model is not None:
        save_path = Path(args.save_model)
        learn.export(save_path)
        print(f"FastAI learner exported to: {save_path.resolve()}")

    if args.save_torch is not None:
        torch_path = Path(args.save_torch)
        torch.save(learn.model.state_dict(), torch_path)
        print(f"PyTorch state_dict saved to: {torch_path.resolve()}")


if __name__ == "__main__":
    main()