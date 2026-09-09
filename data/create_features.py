import chess
import pandas as pd
from tqdm import tqdm


INPUT_PATH = "data/processed/labeled_positions.csv"
OUTPUT_PATH = "data/processed/features.csv"


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board):
    # White material minus Black material.
    balance = 0

    for piece_type, value in PIECE_VALUES.items():
        balance += len(board.pieces(piece_type, chess.WHITE)) * value
        balance -= len(board.pieces(piece_type, chess.BLACK)) * value

    return balance


def mobility(board):
    # Number of legal moves available to the side to move.
    return board.legal_moves.count()


def king_safety(board):
    # Number of friendly pieces protecting the king.
    king_square = board.king(board.turn)

    if king_square is None:
        return 0

    attackers = board.attackers(board.turn, king_square)

    return len(attackers)


def pawn_structure(board):
    # Number of isolated pawns for the side to move.
    color = board.turn
    pawns = board.pieces(chess.PAWN, color)

    isolated = 0

    for square in pawns:
        file = chess.square_file(square)

        left = board.pieces(chess.PAWN, color) & chess.BB_FILES[max(0, file - 1)]
        right = board.pieces(chess.PAWN, color) & chess.BB_FILES[min(7, file + 1)]

        if not left and not right:
            isolated += 1

    return isolated


def game_phase(board):
    # Estimate game phase from remaining non-pawn material.
    phase = 0

    for piece_type in [
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ]:
        phase += len(board.pieces(piece_type, chess.WHITE))
        phase += len(board.pieces(piece_type, chess.BLACK))

    return phase


def is_capture(board, move):
    # Return 1 if the move captures a piece, otherwise 0.
    return int(board.is_capture(move))


def is_check(board, move):
    # Return 1 if the move gives check, otherwise 0.
    board.push(move)
    result = int(board.is_check())
    board.pop()

    return result


def piece_development(board):
    # Number of knights and bishops that left their starting squares.
    starting_squares = [
        (chess.B1, chess.WHITE),
        (chess.G1, chess.WHITE),
        (chess.C1, chess.WHITE),
        (chess.F1, chess.WHITE),
        (chess.B8, chess.BLACK),
        (chess.G8, chess.BLACK),
        (chess.C8, chess.BLACK),
        (chess.F8, chess.BLACK),
    ]

    developed = 0

    for square, color in starting_squares:
        piece = board.piece_at(square)

        if piece is None or piece.color != color:
            developed += 1

    return developed


def squares_controlled_after_move(board, move):
    # Number of squares controlled by the moved piece.
    board.push(move)

    controlled = len(board.attacks(move.to_square))

    board.pop()

    return controlled


def create_features():
    df = pd.read_csv(INPUT_PATH)

    rows = []
    positions_processed = 0
    candidate_moves_generated = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Creating features",
    ):
        board = chess.Board(row["fen"])
        best_move = row["best_move_uci"]

        for move in board.legal_moves:
            candidate_moves_generated += 1

            features = {
                "position_id": positions_processed,
                "fen": row["fen"],
                "candidate_move_uci": move.uci(),
                "material_balance": material_balance(board),
                "mobility": mobility(board),
                "king_safety": king_safety(board),
                "pawn_structure": pawn_structure(board),
                "game_phase": game_phase(board),
                "is_capture": is_capture(board, move),
                "is_check": is_check(board, move),
                "piece_development": piece_development(board),
                "squares_controlled": squares_controlled_after_move(board, move),
                "label": int(move.uci() == best_move),
            }

            rows.append(features)

        positions_processed += 1

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_PATH, index=False)

    positive_labels = features_df["label"].sum()
    negative_labels = len(features_df) - positive_labels

    feature_columns = [
        "material_balance",
        "mobility",
        "king_safety",
        "pawn_structure",
        "game_phase",
        "is_capture",
        "is_check",
        "piece_development",
        "squares_controlled",
    ]

    print(f"\nPositions processed: {positions_processed:,}")
    print(f"Candidate moves generated: {candidate_moves_generated:,}")
    print(f"Positive labels: {positive_labels:,}")
    print(f"Negative labels: {negative_labels:,}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_features()
