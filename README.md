#  Chess

A CNN trained to evaluate chess positions, served through an API, with a simple
web frontend to play against it.

Given a board position, the model predicts how good the position is for the
side to move. To pick a move, every legal move is played out, the resulting
position is scored, and the best-scoring one is returned.

Note: this is a baseline model trained on a small sample of games (~110k
positions). It plays at a weak level and has no lookahead, see
"Known limitations" below before expecting strong play.

## Folder structure

```
chess/
├── .github/workflows/ci.yml   CI pipeline (lint, typecheck, test)
├── api/
│   └── main.py                FastAPI app that serves the model
├── data/
│   ├── raw/                   original game data
│   └── processed/             labeled dataset used for training
├── model/
│   ├── src/
│   │   ├── dataset.py         FEN -> tensor conversion, PyTorch Dataset
│   │   ├── architecture.py    the CNN (ChessCNN)
│   │   ├── train_baseline.py  training script
│   │   └── evaluate.py        Stockfish comparison script
│   ├── models/                saved checkpoints (chess_cnn.pt)
│   └── notebooks/             data pipeline notebook
├── tests/                     API tests
├── Dockerfile                 container for deploying the API
├── Makefile                   shortcuts for common commands
├── requirements.txt           runtime dependencies
└── requirements-dev.txt       lint, typecheck, and test dependencies
```

## The model

- Input: a chess position, encoded as a 13x8x8 tensor (12 planes for piece
  type/color, 1 plane for side to move)
- Output: a single score for that position (squashed to -1..1 with tanh)
- Architecture: a small CNN (3 conv layers + 2 fully connected layers),
  defined in `model/src/architecture.py`
- Training data: real games, labeled with Stockfish's evaluation of the
  position after each move played

To retrain:

```
python model/src/train_baseline.py --data data/processed/chess_cnn_dataset.parquet --epochs 10
```

To check how good the model's move choices actually are, compared to Stockfish:

```
python model/src/evaluate.py --checkpoint model/models/chess_cnn.pt --n-samples 200
```

## The API

`api/main.py`, built with FastAPI. Loads the checkpoint once at startup.

- `GET /health`: basic liveness check
- `POST /predict-move`: send `{"fen": "..."}`, get back the model's chosen
  move (UCI and SAN) plus its predicted evaluation in centipawns

Run it locally:

```
uvicorn api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` to try it directly.

## The frontend

`index.html` is a single self-contained file (HTML/CSS/JS, no build step,
no framework). Open it directly in a browser, or host it as a static file
anywhere. It talks to the API over `fetch()`, so the API must be reachable
from wherever the page is opened (and must have CORS enabled, which it is
by default in `api/main.py`).

Update the `API_BASE_URL` constant near the top of the `<script>` section
if you're pointing it at a different API URL than the one it currently has.

## Setup

```
git clone https://github.com/anuuragg/chess.git
cd chess
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed for lint/typecheck/test
```

Run the API:

```
uvicorn api.main:app --reload
```

Run the tests:

```
pytest tests/
```

Open `index.html` in a browser to play.

## Known limitations

- Small training set (~110k positions from ~4,000 games), not enough data
  for the CNN to learn strong tactical patterns
- No lookahead: the model scores a position in isolation, with no search,
  so it can miss anything that depends on what happens a few moves ahead
- Evaluated against Stockfish: roughly 21% exact move match, average ~240
  centipawn loss per move compared to Stockfish's best choice, weaker than
  a casual human player

The `data/` pipeline that generated the training set was not re-run at full
scale due to local compute constraints. The scripts for it still exist if
you want to regenerate or expand the dataset yourself.