import os

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/features.csv"
BASE_DIR = "data"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def create_split():
    # Find the next available dataset version
    version = 1

    while os.path.exists(f"{BASE_DIR}/v{version}"):
        version += 1

    output_dir = f"{BASE_DIR}/v{version}"
    os.makedirs(output_dir)

    print(f"Creating dataset version: v{version}")

    df = pd.read_csv(INPUT_PATH)

    # Get unique positions
    positions = df["position_id"].unique()

    train_positions, test_positions = train_test_split(
        positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_positions = set(train_positions)
    test_positions = set(test_positions)

    # Split by position_id to prevent data leakage
    train_df = df[df["position_id"].isin(train_positions)].copy()
    test_df = df[df["position_id"].isin(test_positions)].copy()

    # Verify that no position exists in both sets
    overlap = train_positions & test_positions

    assert len(overlap) == 0, "Data leakage detected!"

    # Save as Parquet
    train_df.to_parquet(
        f"{output_dir}/train.parquet",
        index=False,
    )

    test_df.to_parquet(
        f"{output_dir}/test.parquet",
        index=False,
    )

    # Summary
    print("\nDataset split complete!")
    print("\nSplit Summary")
    print("-------------")

    print(f"Train rows:           {len(train_df):,}")
    print(f"Test rows:            {len(test_df):,}")

    print(f"Train positions:      {train_df['position_id'].nunique():,}")
    print(f"Test positions:       {test_df['position_id'].nunique():,}")

    print("\nTrain labels")
    print(f"Positive:             {int(train_df['label'].sum()):,}")
    print(f"Negative:             {int((train_df['label'] == 0).sum()):,}")

    print("\nTest labels")
    print(f"Positive:             {int(test_df['label'].sum()):,}")
    print(f"Negative:             {int((test_df['label'] == 0).sum()):,}")

    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    create_split()
