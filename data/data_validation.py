import chess
import pandas as pd


INPUT_PATH = "data/processed/features.csv"


NUMERIC_FEATURES = [
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


def validate_dataset():
    df = pd.read_csv(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")
    print("\nRunning validation...\n")

    # 1. No null values
    null_count = int(df.isnull().sum().sum())
    assert null_count == 0, f"Found {null_count} null values"

    # 2. Every FEN parses
    invalid_fens = 0

    for fen in df["fen"]:
        try:
            chess.Board(fen)
        except ValueError:
            invalid_fens += 1

    assert invalid_fens == 0, f"Found {invalid_fens} invalid FENs"

    # 3. Every candidate move is legal for its FEN
    invalid_moves = 0

    for _, row in df.iterrows():
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["candidate_move_uci"])

        if move not in board.legal_moves:
            invalid_moves += 1

    assert invalid_moves == 0, f"Found {invalid_moves} illegal candidate moves"

    # 4. Labels are only 0 or 1
    invalid_labels = int((~df["label"].isin([0, 1])).sum())

    assert invalid_labels == 0, f"Found {invalid_labels} invalid labels"

    # 5. Numeric feature columns contain valid values
    non_numeric = []

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)

    assert not non_numeric, f"Non-numeric feature columns found: {non_numeric}"

    invalid_numeric = int(df[NUMERIC_FEATURES].isnull().sum().sum())

    assert invalid_numeric == 0, f"Found {invalid_numeric} invalid numeric values"

    # 6. Exactly one positive label per position
    positives_per_position = df.groupby("position_id")["label"].sum()

    invalid_positions = int((positives_per_position != 1).sum())

    assert invalid_positions == 0, (
        f"Found {invalid_positions} positions " "without exactly one positive label"
    )

    # 7. Validation summary
    print("Validation passed!")

    print("\nValidation Summary")
    print("------------------")
    print(f"Rows:                 {len(df):,}")
    print(f"Positions:            {df['position_id'].nunique():,}")
    print(f"Null values:          {null_count}")
    print(f"Invalid FENs:         {invalid_fens}")
    print(f"Illegal moves:        {invalid_moves}")
    print(f"Invalid labels:       {invalid_labels}")
    print(f"Positive labels:      {int(df['label'].sum()):,}")
    print(f"Negative labels:      {int((df['label'] == 0).sum()):,}")
    print(f"Positions checked:    {len(positives_per_position):,}")
    print(f"Feature count:        {len(NUMERIC_FEATURES)}")


if __name__ == "__main__":
    validate_dataset()
