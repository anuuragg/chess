import pandas as pd
import chess
import chess.pgn
from io import StringIO

INPUT_PATH = "data/processed/chess_games_filtered.csv"
OUTPUT_PATH = "data/processed/positions.csv"

games_df = pd.read_csv(INPUT_PATH)

positions = []


for _, game_row in games_df.iterrows():

    pgn = StringIO(game_row["AN"])
    game = chess.pgn.read_game(pgn)

    board = game.board()

    for move in game.mainline_moves():

        positions.append(
            {
                "fen": board.fen(),
                "played_move_uci": move.uci(),
            }
        )

        board.push(move)


positions_df = pd.DataFrame(positions)

positions_df.to_csv(OUTPUT_PATH, index=False)


print(f"Games processed: {len(games_df):,}")
print(f"Positions extracted: {len(positions_df):,}")
print(f"Saved to: {OUTPUT_PATH}")
