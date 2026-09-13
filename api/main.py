import importlib
import sys
from pathlib import Path

import chess
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "model" / "src"))

architecture = importlib.import_module("architecture")
dataset_module = importlib.import_module("dataset")

ChessCNN = architecture.ChessCNN
board_to_tensor = dataset_module.board_to_tensor
unscale_eval = dataset_module.unscale_eval

CHECKPOINT_PATH = ROOT_DIR / "model" / "models" / "chess_cnn.pt"

app = FastAPI(title="Chess Move Predictor")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessCNN()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()


class PredictRequest(BaseModel):
    fen: str


class PredictResponse(BaseModel):
    move_uci: str
    move_san: str
    predicted_eval_cp: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-move", response_model=PredictResponse)
def predict_move(request: PredictRequest):
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid FEN")

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise HTTPException(status_code=400, detail="no legal moves for this position")

    side_to_move_is_white = board.turn == chess.WHITE
    best_move: chess.Move | None = None
    best_move_san: str | None = None
    best_score: float | None = None

    for move in legal_moves:
        move_san = board.san(move)
        board.push(move)
        tensor = board_to_tensor(board).reshape(1, 13, 8, 8)
        board.pop()

        with torch.no_grad():
            pred = model(torch.from_numpy(tensor).to(device)).item()

        is_better = (
            best_score is None
            or (side_to_move_is_white and pred > best_score)
            or (not side_to_move_is_white and pred < best_score)
        )
        if is_better:
            best_score = pred
            best_move = move
            best_move_san = move_san

    assert best_move is not None
    assert best_move_san is not None
    assert best_score is not None

    return PredictResponse(
        move_uci=best_move.uci(),
        move_san=best_move_san,
        predicted_eval_cp=round(unscale_eval(best_score), 1),
    )
