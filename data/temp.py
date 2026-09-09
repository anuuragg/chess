import chess
import chess.engine
import pandas as pd
import time

INPUT_PATH = "data/processed/positions.csv"
OUTPUT_PATH = "data/processed/labeled_positions.csv"
STOCKFISH_PATH = "/usr/games/stockfish"

df = pd.read_csv(INPUT_PATH)

# Filter 2650+ average Elo
df["AverageElo"] = (df["WhiteElo"] + df["BlackElo"]) / 2
df = df[df["AverageElo"] >= 2650].copy()

total_positions = len(df)

# Only benchmark the first 10 positions
sample = df.head(10).copy()

print(f"Total 2650+ positions: {total_positions:,}")
print(f"Benchmarking: {len(sample)} positions")

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

best_moves = []
best_evals = []

start_time = time.time()

for _, row in sample.iterrows():
    board = chess.Board(row["fen"])

    result = engine.analyse(board, chess.engine.Limit(depth=16))

    best_moves.append(result["pv"][0].uci())
    best_evals.append(result["score"].pov(board.turn).score(mate_score=100000))

elapsed = time.time() - start_time

engine.quit()

# Estimate full processing time
avg_time = elapsed / len(sample)
estimated_seconds = avg_time * total_positions

sample["best_move_uci"] = best_moves
sample["best_eval"] = best_evals

print(f"\nBenchmark time: {elapsed:.2f} seconds")
print(f"Average time/position: {avg_time:.2f} seconds")
print(f"Estimated total time: {estimated_seconds / 3600:.2f} hours")
print(f"Estimated total time: {estimated_seconds / 86400:.2f} days")
