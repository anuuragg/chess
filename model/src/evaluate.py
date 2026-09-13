import argparse
import random

import chess
import chess.engine
import pandas as pd
import torch
from architecture import ChessCNN
from dataset import board_to_tensor, unscale_eval

STOCKFISH_PATH = "/usr/games/stockfish"


def load_model(checkpoint_path, device):
    model = ChessCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def model_pick_move(model, board, device):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = None

    for move in legal_moves:
        board.push(move)
        tensor = board_to_tensor(board).reshape(1, 13, 8, 8)
        board.pop()

        with torch.no_grad():
            pred = model(torch.from_numpy(tensor).to(device)).item()

        if board.turn == chess.WHITE:
            is_better = best_score is None or pred > best_score
        else:
            is_better = best_score is None or pred < best_score

        if is_better:
            best_score = pred
            best_move = move

    return best_move, unscale_eval(best_score)


def stockfish_eval(engine, board, depth):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(chess.WHITE)
    if score.is_mate():
        return 10000 if score.mate() > 0 else -10000
    return score.score()


def stockfish_best_move(engine, board, depth):
    result = engine.play(board, chess.engine.Limit(depth=depth))
    return result.move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/chess_cnn_dataset.parquet")
    parser.add_argument("--checkpoint", default="model/models/chess_cnn.pt")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.data)
    unique_fens = df["fen"].drop_duplicates().tolist()
    sampled_fens = random.sample(unique_fens, min(args.n_samples, len(unique_fens)))

    model = load_model(args.checkpoint, device)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    top1_matches = 0
    centipawn_losses = []

    for i, fen in enumerate(sampled_fens):
        board = chess.Board(fen)

        model_move, _ = model_pick_move(model, board, device)

        sf_best_move = stockfish_best_move(engine, board.copy(), args.depth)

        board_after_sf = board.copy()
        board_after_sf.push(sf_best_move)
        sf_best_eval = stockfish_eval(engine, board_after_sf, args.depth)

        board_after_model = board.copy()
        board_after_model.push(model_move)
        model_move_eval = stockfish_eval(engine, board_after_model, args.depth)

        if model_move == sf_best_move:
            top1_matches += 1

        loss = (
            (sf_best_eval - model_move_eval)
            if board.turn == chess.WHITE
            else (model_move_eval - sf_best_eval)
        )
        centipawn_losses.append(max(0, loss))

        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sampled_fens)} positions evaluated")

    engine.quit()

    top1_accuracy = top1_matches / len(sampled_fens)
    avg_centipawn_loss = sum(centipawn_losses) / len(centipawn_losses)

    print(f"positions evaluated: {len(sampled_fens)}")
    print(f"top-1 move match rate: {top1_accuracy:.4f}")
    print(f"average centipawn loss vs stockfish best: {avg_centipawn_loss:.1f}")


if __name__ == "__main__":
    main()
