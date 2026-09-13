import argparse

import pandas as pd
import torch
from architecture import ChessCNN
from dataset import ChessPositionDataset, assign_game_ids
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader


def weighted_mse(preds, targets, weights):
    per_sample_loss = torch.nn.functional.mse_loss(preds, targets, reduction="none")
    return (per_sample_loss * weights).mean()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_samples = 0

    for tensors, targets, weights in loader:
        tensors = tensors.to(device)
        targets = targets.to(device)
        weights = weights.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(tensors)
            loss = weighted_mse(preds, targets, weights)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = tensors.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/chess_cnn_dataset.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out", default="model/models/chess_cnn.pt")
    parser.add_argument("--no-weighting", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df["game_id"] = assign_game_ids(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["game_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    assert set(train_df["game_id"]).isdisjoint(set(val_df["game_id"]))
    print(f"train rows: {len(train_df)}, val rows: {len(val_df)}")
    print(f"val eval_after stats:\n{val_df['eval_after'].describe()}")

    use_weighting = not args.no_weighting
    train_dataset = ChessPositionDataset(train_df, use_weighting=use_weighting)
    val_dataset = ChessPositionDataset(val_df, use_weighting=use_weighting)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"epoch {epoch}/{args.epochs} - train_loss {train_loss:.4f} - val_loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)

    print(f"best val loss: {best_val_loss:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
