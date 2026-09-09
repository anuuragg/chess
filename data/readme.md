## Data Pipeline

The original dataset contained **6M+ chess games**, which was too large to process efficiently in a single pass. The pipeline therefore filters the source games based on **average Elo ≥ 2650** before further processing.

The filtered games are then processed as follows:

```text
Raw chess games (6M+)
        ↓
Filter games by average Elo ≥ 2650
        ↓
Replay AN move sequences using python-chess
        ↓
Generate FEN for the position before each move
        ↓
Save positions
        ↓
Run Stockfish analysis
        ↓
Generate move-quality labels
```

### Steps

1. **Filter the source games**

   * Select games with an average Elo of **2650+** to retain higher-quality games while keeping the dataset computationally manageable.

2. **Generate positions**

   * Replay the games using `python-chess`.
   * Generate a **FEN** representation of the board position before every move.

3. **Stockfish analysis**

   * Analyze the generated positions using Stockfish.
   * Compare the played move against Stockfish's recommended move.

4. **Generate labels**

   * Use the Stockfish analysis to label positions according to the quality of the played move.

### Dataset source

The original raw dataset is not included in the repository because of its size. It can be downloaded from the original Kaggle source:

**Source:** [Kaggle - Chess Games Dataset](https://www.kaggle.com/datasets/arevel/chess-games/data)