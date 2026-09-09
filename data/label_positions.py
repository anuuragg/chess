import chess
import chess.engine
import pandas as pd
from tqdm import tqdm
import os

INPUT_PATH = "data/processed/positions.csv"
OUTPUT_PATH = "data/processed/labeled_positions.csv"

CHECKPOINT_INTERVAL = 500

df = pd.read_csv(INPUT_PATH)

total = len(df)

# Resume from checkpoint
if os.path.exists(OUTPUT_PATH):
    labeled = pd.read_csv(OUTPUT_PATH)
    start = len(labeled)
    print(f"Resuming from {start:,}/{total:,}")
else:
    labeled = pd.DataFrame()
    start = 0
    print(f"Starting from 0/{total:,}")

# Start Stockfish
engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

# Limit CPU usage
engine.configure({"Threads": 1})

try:
    for i in tqdm(
        range(start, total), initial=start, total=total, desc="Analyzing positions"
    ):
        row = df.iloc[i].copy()

        board = chess.Board(row["fen"])

        result = engine.analyse(board, chess.engine.Limit(depth=12))

        row["best_move_uci"] = result["pv"][0].uci()
        row["best_eval"] = result["score"].pov(board.turn).score(mate_score=100000)

        labeled = pd.concat([labeled, pd.DataFrame([row])], ignore_index=True)

        # Save checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            labeled.to_csv(OUTPUT_PATH, index=False)

            print(f"\nCheckpoint saved: {i + 1:,}/{total:,}")

finally:
    engine.quit()

# Final save
labeled.to_csv(OUTPUT_PATH, index=False)

agreement = (labeled["played_move_uci"] == labeled["best_move_uci"]).mean() * 100

print(f"\nPositions processed: {len(labeled):,}")
print(f"Played move == best move: {agreement:.2f}%")
print(f"Saved to: {OUTPUT_PATH}")
