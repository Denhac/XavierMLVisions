# Contributing

Build a game, add features, fix bugs. Make it yours.

## System Overview

Tile screensaver → hand detected → game selection → object detected → game launches. Your game plugs into this pipeline.

## Current Games

- **TileShuffle** — screensaver / idle display
- **SimpleHunt** — object hunt using hand detection
- **DemonQuest** — projection game with animated sequences
- **TrainClassifier** — educational: visitors train a YOLO classifier live on-device

Games can be interactive, educational, or both.

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

You can train your own models on the Xavier's GPU for use in your game. See `games/TrainClassifier/` for an example that trains a YOLO classifier on-device.

## Testing on Xavier

Xavier is on the denhac local network only — no remote access.

```bash
ssh colin@10.11.3.65
cd ~/MLVisions
git fetch origin && git checkout your-feature
```

## Pushing from Xavier

The Xavier has a deploy key with write access to GitHub. Anyone at denhac can commit and push directly from it. Set your identity before committing:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
git add -A && git commit -m "your message"
git push origin master
```

## Before You Push

Merge master into your branch and make sure nothing is broken:

```bash
git fetch origin
git merge origin/master
# resolve any conflicts
# test that existing games still run
git push
```

Do this before opening a PR. If master has changed since you branched, your PR should already include those changes.

## Rules

- Don't break other games
- Exit on Q (keyboard) or when an exit class is detected (camera)
- Don't commit `.pyc`, `venv/`, IDE configs, or large binaries (use Git LFS)
