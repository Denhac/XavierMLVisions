# Contributing

Build a game, add features, fix bugs. Make it yours.

## System Overview

Tile screensaver → hand detected → game selection → object detected → game launches. Your game plugs into this pipeline.

## New Game Structure

```
games/YourGame/
├── game.py
├── README.md
└── requirements.txt
```

## Workflow

1. Clone: `git clone git@github.com:Denhac/XavierMLVisions.git`
2. Branch: `git checkout -b your-feature`
3. Develop and test locally (CPU inference works)
4. Push and open a PR
5. At denhac, pull to Xavier and test with GPU + projector
6. Merge

## Training Models

You can train your own models on the Xavier's GPU for use in your game. See `games/TrainClassifier/` and `games/TrainTheAI/` for examples of games that train YOLO classifiers on-device.

## Testing on Xavier

Xavier is on the denhac local network only — no remote access.

```bash
ssh colin@10.11.3.65
cd ~/MLVisions
git fetch origin && git checkout your-feature
```

## Rules

- Don't break other games
- Exit on Q (keyboard) or when an exit class is detected (camera)
- Don't commit `.pyc`, `venv/`, IDE configs, or large binaries (use Git LFS)
