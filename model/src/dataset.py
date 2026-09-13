import chess
import numpy as np
import torch
from torch.utils.data import Dataset

PIECE_TO_PLANE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def board_to_tensor(board):
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = PIECE_TO_PLANE[piece.symbol()]
        tensor[plane, row, col] = 1.0
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    return tensor


def scale_eval(cp, clip=1000, scale=400):
    cp = max(-clip, min(clip, cp))
    return np.tanh(cp / scale)


def unscale_eval(scaled, scale=400):
    return np.arctanh(np.clip(scaled, -0.999999, 0.999999)) * scale


def assign_game_ids(df):
    game_ids = []
    current_id = 0
    prev_phase = -1
    for phase in df["game_phase"]:
        if phase < prev_phase:
            current_id += 1
        game_ids.append(current_id)
        prev_phase = phase
    return game_ids


def eval_weight(cp, near_equal_band=100, decisive_weight=3.0):
    return 1.0 if abs(cp) <= near_equal_band else decisive_weight


class ChessPositionDataset(Dataset):
    def __init__(self, df, use_weighting=True):
        self.fens = df["fen"].values
        self.moves = df["move_uci"].values
        self.evals = df["eval_after"].values
        self.use_weighting = use_weighting

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        board.push(chess.Move.from_uci(self.moves[idx]))
        tensor = board_to_tensor(board)
        target = scale_eval(self.evals[idx])
        weight = eval_weight(self.evals[idx]) if self.use_weighting else 1.0
        return (
            torch.from_numpy(tensor),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("data/processed/chess_cnn_dataset.parquet")
    dataset = ChessPositionDataset(df)

    print(len(dataset))
    sample_tensor, sample_target, sample_weight = dataset[0]
    print(sample_tensor.shape)
    print(sample_target)
    print(sample_weight)

    board = chess.Board()
    check_tensor = board_to_tensor(board)
    assert check_tensor[PIECE_TO_PLANE["P"], 6, 4] == 1.0
    assert check_tensor[PIECE_TO_PLANE["k"], 0, 4] == 1.0
    assert check_tensor[12, 0, 0] == 1.0
    print("tensor orientation correct")
