import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from fastai.vision.all import Learner, accuracy, load_learner

# Import model & dataloader builders from your training script
from darkcovidnet_3class import build_darkcovidnet, get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DarkCovidNet 3-class model on a validation set."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Path to the root dataset folder. "
            "Inside it must contain 'train' and 'valid' subfolders. "
            "Example: /path/to/thyroid_dataset_more_classes_full_3class"
        ),
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader (default: 4).",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-pkl",
        type=str,
        help=(
            "Path to a FastAI exported learner (.pkl), "
            "e.g. darkcovidnet_3class.pkl"
        ),
    )
    group.add_argument(
        "--model-state",
        type=str,
        help=(
            "Path to a raw PyTorch state_dict (.pth), "
            "e.g. darkcovidnet_3class_state.pth"
        ),
    )

    return parser.parse_args()


def evaluate_learner(learn: Learner):
    """Run evaluation on the learner's validation dataloader."""
    dls = learn.dls
    print("Number of validation examples:", len(dls.valid_ds))

    # Get predictions and targets
    probs, targets = learn.get_preds(dl=dls.valid)
    acc = accuracy(probs, targets)

    # Convert accuracy to float for printing
    acc_float = float(acc)
    print(f"fastai accuracy: {acc_float:.4f}")

    # Manual accuracy, confusion matrix, classification report
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


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data_dir {data_dir} does not exist. "
            f"Please check the --data-dir argument."
        )

    print(f"Using data directory: {data_dir.resolve()}")

    # Build dataloaders for evaluation
    dls = get_dataloaders(
        data_dir=data_dir,
        bs=args.bs,
        num_workers=args.num_workers,
    )

    if args.model_pkl is not None:
        # 1) Load FastAI exported learner
        model_path = Path(args.model_pkl)
        if not model_path.exists():
            raise FileNotFoundError(f"model-pkl {model_path} does not exist.")

        print(f"Loading FastAI learner from: {model_path.resolve()}")
        learn = load_learner(model_path)

        # Override the dataloaders with current dataset (optional but recommended)
        learn.dls = dls

    elif args.model_state is not None:
        # 2) Load raw PyTorch state_dict into a fresh model
        state_path = Path(args.model_state)
        if not state_path.exists():
            raise FileNotFoundError(f"model-state {state_path} does not exist.")

        print(f"Loading PyTorch state_dict from: {state_path.resolve()}")

        num_classes = len(dls.vocab)
        model = build_darkcovidnet(num_classes=num_classes)

        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # Wrap model in a FastAI Learner for easy evaluation
        learn = Learner(
            dls,
            model,
            loss_func=nn.CrossEntropyLoss(),
            metrics=accuracy,
        )

    else:
        raise ValueError("You must provide either --model-pkl or --model-state.")

    # Run evaluation
    evaluate_learner(learn)


if __name__ == "__main__":
    main()