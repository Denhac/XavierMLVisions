# Train The AI - Educational ML Experience with REAL Training

An interactive game that teaches machine learning by **actually training a model** on visitor photos.

**This is NOT a simulation** - visitors capture photos, the AI genuinely trains on them (~30-60 sec on Xavier GPU), and they test THEIR trained model.

## Quick Start (Xavier)

```bash
cd ~/MLVisions
python3 games/TrainTheAI/game.py
```

With different camera:
```bash
python3 games/TrainTheAI/game.py --source 1
```

## Game Flow (~3-4 minutes total)

### Phase 1: CAPTURE OBJECT (~10 sec)
- Visitor picks ANY object (phone, keys, water bottle, hand...)
- Press SPACE → 10-second auto-recording starts
- Move object around for different angles (50 frames captured)

### Phase 2: CAPTURE BACKGROUND (~8 sec)
- Visitor REMOVES object from view
- Press SPACE → 8-second auto-recording starts
- Camera captures what ISN'T the object (40 frames)

### Phase 3: EDUCATION SLIDES (~30 sec)
- 5 arrow-through slides explaining:
  - What is a neural network?
  - What are weights? (with matrix example)
  - What is transfer learning?
  - What is loss? (with formula)
  - Training config summary
- Use LEFT/RIGHT arrows to navigate
- Press SPACE on last slide to start training

### Phase 4: REAL TRAINING (~30-60 sec)
- **Actual YOLOv8n-cls fine-tuning** on their photos
- Shows REAL epoch count, loss values
- Technical facts rotate during training
- Training runs on Xavier GPU

### Phase 5: PLAY (45 sec)
- Test THEIR trained vision classifier
- Show object to camera, watch confidence
- Hold steady for 1.2 sec to score
- Uses the model they just trained!

### Phase 6: RESULTS
- Final score + training accuracy
- Educational summary of what they experienced

## Controls

| Key | Action |
|-----|--------|
| SPACE | Start recording / Continue / Start training |
| ←/→ | Navigate education slides |
| R | Reset to title screen |
| Q | Quit |

## Educational Talking Points

### Before they play:
> "You're about to do what real ML engineers do every day -
> train an AI from scratch on your own photos!"

### During object capture:
> "Each photo you take becomes 'training data.' The more angles
> and lighting conditions you capture, the better the AI will learn."

### During background capture:
> "The AI needs to learn what ISN'T your object too!
> This is called 'negative examples' - it's how AI learns the difference."

### During REAL training:
> "This is ACTUALLY happening right now! The AI is adjusting
> 3 million numbers called 'weights' to match your photos.
> That 'loss' number dropping means it's getting smarter!"

### During play:
> "This is YOUR model - trained on YOUR photos just now!
> The confidence shows how sure it is. Notice how angles
> and lighting affect recognition - that's why training data matters!"

### Key takeaways:
1. **AI learns from YOUR examples** - they actually trained this model
2. **Negative examples matter** - background photos teach "what it's NOT"
3. **Transfer learning** - started from a model that knows 1000 objects
4. **Real ML pipeline** - capture → train → test (just like production)

## Technical Notes

- **REAL training** - Uses `YOLO.train()` with visitor's actual photos
- **Transfer learning** - Fine-tunes yolov8n-cls.pt (pretrained on ImageNet 1000 classes)
- **5 epochs** - Quick training (~30-60 sec on Xavier GPU)
- **~90 training images** - 50 object + 40 background frames
- **Binary classification** - Their object class vs background class
- **Model**: YOLOv8n-cls (~3M parameters, 224x224 input)
- Training runs in background thread while UI shows progress
- Educational slides explain: neural networks, weights, transfer learning, loss functions

## Files

```
games/TrainTheAI/
    game.py     - Main game (this file)
    README.md   - This documentation
```

## Troubleshooting

**Camera not opening:**
```bash
# Check available cameras
ls /dev/video*

# Try different source
python3 games/TrainTheAI/game.py --source 1
```

**Model not found:**
```bash
# Verify model exists
ls -la ~/MLVisions/models/7class_v1/best.onnx
```

**Display issues:**
- Make sure you're on local console, not SSH
- Or use `ssh -X xavier` for X11 forwarding
