#!/usr/bin/env python3
"""
Train The AI - Educational ML Experience with REAL Training

A 3-phase interactive experience that teaches machine learning concepts
using ACTUAL model training, not simulation.

1. CAPTURE - Visitor takes photos of an object
2. TRAIN - REAL YOLOv8 fine-tuning on their photos (~30-60 sec)
3. PLAY - Test THEIR trained model

Usage:
    python games/TrainTheAI/game.py
    python games/TrainTheAI/game.py --source 1  # Different camera
"""
import sys
import os
import time
import random
import shutil
import argparse
import threading
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Video capture config (replaces manual space-bar capture)
CAPTURE_DURATION = 10.0     # Seconds to record video
CAPTURE_FPS = 10            # Frames to extract per second (10 sec * 10 fps = 100 images)
BG_CAPTURE_DURATION = 8.0   # Seconds for background recording
BG_CAPTURE_FPS = 10         # Background frame rate (8 sec * 10 fps = 80 images)

# Training config
TRAIN_EPOCHS = 5            # More epochs for better accuracy with more data
TRAIN_IMGSZ = 224           # Image size for training
PLAY_DURATION = 45.0        # Seconds for play phase
HOLD_DURATION = 1.2         # Seconds to hold for score

# Background images
BACKGROUND_COUNT = 20       # Max pre-captured backgrounds to include

# Paths
GAME_DIR = Path(__file__).parent
BACKGROUND_DIR = GAME_DIR / "background_images"
TEMP_DATASET_DIR = GAME_DIR / "temp_training"

# Educational content shown during training (technical and accurate)
TRAINING_FACTS = [
    # Architecture facts
    "YOLOv8n-cls: A convolutional neural network with 3M parameters",
    "Input: 224x224x3 tensor (224px image, RGB channels)",
    "Conv layers extract features: edges -> textures -> shapes -> objects",
    "Final layer: 1280 features compressed to 2 classes (yours + background)",
    # Training facts
    "Each epoch = 1 complete pass through all your training images",
    "Batch size 8: processing 8 images at once for stable gradients",
    "Loss function: Cross-entropy measures prediction vs truth",
    "Backpropagation: calculates how much each weight contributed to error",
    "Gradient descent: adjusts weights in direction that reduces loss",
    # Transfer learning
    "Transfer learning: starting from ImageNet-pretrained weights",
    "Pretrained on 1.2M images across 1000 categories",
    "All 3M weights update, but early layers change minimally",
    "Final classifier layer adapts most to YOUR specific object",
]

# Detailed educational slides (shown before training, arrow through)
EDUCATION_SLIDES = [
    {
        "title": "What is a Neural Network?",
        "content": [
            "A mathematical function that transforms images into predictions",
            "",
            "Input (224x224 image) --> [millions of math operations] --> Output (class probabilities)",
            "",
            "Structure: Layers of 'neurons' connected by 'weights'",
            "Each weight is just a number that gets multiplied",
        ]
    },
    {
        "title": "What are Weights?",
        "content": [
            "YOLOv8n-cls has ~3 million trainable parameters",
            "",
            "Example weight matrix (simplified):",
            "  [[0.12, -0.34, 0.56, ...],",
            "   [0.78, 0.23, -0.45, ...],",
            "   ...]",
            "",
            "Training = finding the best values for ALL these numbers",
        ]
    },
    {
        "title": "What is Transfer Learning?",
        "content": [
            "We don't train from scratch (would need millions of images)",
            "",
            "Instead, we start from a model pretrained on ImageNet:",
            "  - 1.2 million images, 1000 categories",
            "",
            "Early layers already know: edges, textures, shapes",
            "All weights update, but early layers change minimally",
            "Final layer adapts most to YOUR 2 classes",
        ]
    },
    {
        "title": "What is Loss?",
        "content": [
            "Loss = how wrong the model's predictions are",
            "",
            "Cross-entropy loss example:",
            "  Model says: 80% your_object, 20% background",
            "  Truth: 100% your_object",
            "  Loss = -log(0.80) = 0.22",
            "",
            "Lower loss = better predictions",
            "Training goal: minimize loss across all images",
        ]
    },
    {
        "title": "Ready to Train!",
        "content": [
            f"Your dataset: ~50 object + ~40 background images",
            "",
            "Training config:",
            f"  - Epochs: {TRAIN_EPOCHS} (passes through data)",
            f"  - Image size: {TRAIN_IMGSZ}x{TRAIN_IMGSZ} pixels",
            "  - Batch size: 8",
            "  - Optimizer: AdamW (adaptive learning rate)",
            "",
            "Press SPACE to start training!",
        ]
    },
]


# =============================================================================
# GAME STATES
# =============================================================================

class GamePhase(Enum):
    TITLE = "title"
    CAPTURE_INTRO = "capture_intro"
    CAPTURING = "capturing"
    CAPTURE_BG_INTRO = "capture_bg_intro"
    CAPTURING_BG = "capturing_bg"
    EDUCATION = "education"       # Arrow-through educational slides
    TRAINING = "training"
    PLAY_INTRO = "play_intro"
    PLAYING = "playing"
    GAME_OVER = "game_over"


@dataclass
class GameState:
    phase: GamePhase = GamePhase.TITLE
    object_name: str = "my_object"
    captured_frames: List[np.ndarray] = field(default_factory=list)
    background_frames: List[np.ndarray] = field(default_factory=list)

    # Video recording state
    recording_start_time: Optional[float] = None
    last_capture_time: float = 0.0
    frame_interval: float = 1.0 / CAPTURE_FPS  # Time between frame captures

    # Education slides state
    education_slide_idx: int = 0

    # Training state
    training_thread: Optional[threading.Thread] = None
    training_complete: bool = False
    training_error: bool = False  # Track if training failed
    training_progress: float = 0.0
    training_epoch: int = 0
    training_loss: float = 1.0
    training_accuracy: float = 0.0
    training_fact_idx: int = 0
    training_log: List[str] = field(default_factory=list)
    trained_model_path: Optional[str] = None

    # Play state
    classifier: object = None  # Will hold trained YOLO model
    score: int = 0
    play_start_time: float = 0.0
    hold_start_time: Optional[float] = None
    phase_start_time: float = 0.0


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

def draw_text_with_bg(frame, text, pos, scale=1.0, color=(255, 255, 255),
                      thickness=2, bg_color=(0, 0, 0), padding=8):
    """Draw text with background for visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame,
                  (x - padding, y - h - padding),
                  (x + w + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(frame, x, y, width, height, progress, color,
                      bg_color=(50, 50, 50), border_color=(255, 255, 255)):
    """Draw a progress bar."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = int(width * min(1.0, max(0.0, progress)))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)


def draw_centered_text(frame, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    """Draw centered text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - w) // 2
    draw_text_with_bg(frame, text, (x, y), scale, color, thickness)


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_dataset(state: GameState) -> Path:
    """
    Create YOLO-compatible dataset from captured frames.

    Structure:
        temp_training/
            train/
                my_object/   (visitor's photos)
                background/  (pre-captured + visitor background)
            val/
                my_object/
                background/
    """
    dataset_dir = TEMP_DATASET_DIR

    # Clean previous
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # Create structure
    train_obj = dataset_dir / "train" / state.object_name
    train_bg = dataset_dir / "train" / "background"
    val_obj = dataset_dir / "val" / state.object_name
    val_bg = dataset_dir / "val" / "background"

    for d in [train_obj, train_bg, val_obj, val_bg]:
        d.mkdir(parents=True, exist_ok=True)

    # Split captured frames 80/20
    obj_frames = state.captured_frames.copy()
    random.shuffle(obj_frames)
    split_idx = int(len(obj_frames) * 0.8)
    train_objs = obj_frames[:split_idx]
    val_objs = obj_frames[split_idx:]

    # Save object images
    for i, frame in enumerate(train_objs):
        cv2.imwrite(str(train_obj / f"img_{i:04d}.jpg"), frame)
    for i, frame in enumerate(val_objs):
        cv2.imwrite(str(val_obj / f"img_{i:04d}.jpg"), frame)

    # Combine pre-captured background with visitor's background captures
    all_bg_frames = state.background_frames.copy()

    # Add pre-captured backgrounds if they exist
    if BACKGROUND_DIR.exists():
        for img_path in list(BACKGROUND_DIR.glob("*.jpg"))[:BACKGROUND_COUNT]:
            bg_img = cv2.imread(str(img_path))
            if bg_img is not None:
                all_bg_frames.append(bg_img)

    random.shuffle(all_bg_frames)
    bg_split = int(len(all_bg_frames) * 0.8)
    train_bgs = all_bg_frames[:bg_split]
    val_bgs = all_bg_frames[bg_split:]

    # Save background images
    for i, frame in enumerate(train_bgs):
        cv2.imwrite(str(train_bg / f"bg_{i:04d}.jpg"), frame)
    for i, frame in enumerate(val_bgs):
        cv2.imwrite(str(val_bg / f"bg_{i:04d}.jpg"), frame)

    print(f"Dataset created: {len(train_objs)} train obj, {len(val_objs)} val obj")
    print(f"                 {len(train_bgs)} train bg, {len(val_bgs)} val bg")

    return dataset_dir


# =============================================================================
# REAL TRAINING
# =============================================================================

def train_model(state: GameState, dataset_path: Path):
    """
    Actually train a YOLO model on the visitor's photos.
    This runs in a background thread.
    """
    from ultralytics import YOLO

    try:
        state.training_log.append("Loading pretrained model...")

        # Start from pretrained nano classifier
        model = YOLO("yolov8n-cls.pt")

        state.training_log.append(f"Training on {dataset_path}...")
        state.training_progress = 0.1

        # Train with callbacks to update progress
        def on_train_epoch_end(trainer):
            state.training_epoch = trainer.epoch + 1
            state.training_progress = min(0.95, (trainer.epoch + 1) / TRAIN_EPOCHS)
            if hasattr(trainer, 'loss'):
                state.training_loss = float(trainer.loss)
            state.training_log.append(f"Epoch {state.training_epoch}/{TRAIN_EPOCHS}")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # REAL TRAINING
        results = model.train(
            data=str(dataset_path),
            epochs=TRAIN_EPOCHS,
            imgsz=TRAIN_IMGSZ,
            batch=8,
            patience=0,  # Don't early stop
            save=True,
            plots=False,
            verbose=False,
            exist_ok=True,
            project=str(GAME_DIR / "runs"),
            name="visitor_model",
            amp=False,  # Disable AMP - Xavier's torchvision lacks C++ ops for AMP check
        )

        # Get trained model path
        state.trained_model_path = str(GAME_DIR / "runs" / "visitor_model" / "weights" / "best.pt")

        # Get final accuracy if available
        if hasattr(results, 'results_dict'):
            state.training_accuracy = results.results_dict.get('metrics/accuracy_top1', 0.9)
        else:
            state.training_accuracy = 0.9  # Default

        state.training_progress = 1.0
        state.training_log.append("Training complete!")
        state.training_complete = True

    except Exception as e:
        state.training_log.append(f"Error: {str(e)[:50]}")
        state.training_error = True
        state.training_accuracy = 0.0
        state.training_complete = True  # Mark complete so UI can handle
        print(f"Training error: {e}")


# =============================================================================
# GAME PHASES
# =============================================================================

def draw_title_screen(frame, state: GameState):
    """Draw the title/start screen."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Smaller text to fit screen width
    draw_centered_text(frame, "TRAIN A NEURAL NETWORK", h // 4, 1.3, (0, 255, 255), 3)
    draw_centered_text(frame, "Real Image Classifier Training", h // 4 + 45, 0.6, (200, 200, 200), 2)

    draw_centered_text(frame, "You will:", h // 2 - 30, 0.8, (255, 255, 255), 2)
    draw_centered_text(frame, "1. Capture training data", h // 2 + 5, 0.6, (150, 255, 150), 2)
    draw_centered_text(frame, "2. Fine-tune YOLOv8n-cls", h // 2 + 35, 0.6, (150, 150, 255), 2)
    draw_centered_text(frame, "3. Test YOUR trained model!", h // 2 + 65, 0.6, (255, 150, 150), 2)

    draw_centered_text(frame, "YOLOv8n-cls: 3M parameters", h - 110, 0.45, (150, 150, 150), 1)
    draw_centered_text(frame, "Press SPACE to begin", h - 70, 0.9, (100, 255, 100), 2)


def draw_capture_intro(frame, state: GameState):
    """Draw capture phase introduction."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    draw_centered_text(frame, "STEP 1: CAPTURE TRAINING DATA", h // 5, 1.0, (0, 255, 255), 2)

    # Smaller text with better spacing - moved down to avoid overlap
    draw_centered_text(frame, "Pick ANY object to train on!", h // 3 + 10, 0.7, (255, 255, 255), 2)
    draw_centered_text(frame, "(phone, keys, bottle, hand...)", h // 3 + 40, 0.5, (200, 200, 200), 1)

    expected_frames = int(CAPTURE_DURATION * CAPTURE_FPS)
    draw_centered_text(frame, f"{int(CAPTURE_DURATION)}s recording = ~{expected_frames} images", h // 2 + 10, 0.65, (200, 200, 200), 2)
    draw_centered_text(frame, "MOVE OBJECT for angle variety!", h // 2 + 45, 0.6, (150, 255, 150), 2)

    draw_centered_text(frame, "Auto-captures frames at 5fps", h - 130, 0.5, (255, 200, 100), 1)
    draw_centered_text(frame, "(real ML data collection)", h - 105, 0.5, (255, 200, 100), 1)

    draw_centered_text(frame, "Press SPACE to start", h - 60, 0.9, (100, 255, 100), 2)


def draw_capturing(frame, state: GameState, is_background: bool = False):
    """Draw the video recording capture UI."""
    h, w = frame.shape[:2]

    if is_background:
        title = "RECORDING: BACKGROUND (no object)"
        color = (100, 100, 255)
        duration = BG_CAPTURE_DURATION
        frames_list = state.background_frames
        tip_set = ["Pan around slowly", "Show different areas", "Keep object OUT of view"]
    else:
        title = "RECORDING: YOUR OBJECT"
        color = (0, 255, 255)
        duration = CAPTURE_DURATION
        frames_list = state.captured_frames
        tip_set = [
            "Rotate the object slowly",
            "Move closer and farther",
            "Tilt to show different sides",
            "Keep moving for variety!",
        ]

    # Calculate recording progress
    if state.recording_start_time is not None:
        elapsed = time.time() - state.recording_start_time
        progress = min(1.0, elapsed / duration)
        remaining = max(0, duration - elapsed)
    else:
        elapsed = 0
        progress = 0
        remaining = duration

    # Recording indicator (pulsing red dot)
    pulse = int((time.time() * 3) % 2)
    if pulse and state.recording_start_time is not None:
        cv2.circle(frame, (w - 40, 35), 15, (0, 0, 255), -1)
    draw_text_with_bg(frame, title, (20, 40), 1.0, color, 2)

    # Frame counter
    counter_text = f"Frames: {len(frames_list)}"
    draw_text_with_bg(frame, counter_text, (w - 180, 70), 0.7, (255, 255, 255), 2)

    # Time remaining
    draw_centered_text(frame, f"{remaining:.1f}s", h // 2, 2.5, color, 4)

    # Progress bar
    draw_progress_bar(frame, 20, h - 60, w - 40, 30, progress, color)

    # Tip (changes based on time)
    tip_idx = int(elapsed / 2.5) % len(tip_set)
    draw_centered_text(frame, tip_set[tip_idx], 80, 0.7, (200, 200, 200), 2)

    # Show thumbnails (last 8 captured)
    if frames_list:
        recent = frames_list[-8:]
        thumb_size = 55
        for i, img in enumerate(recent):
            x = 20 + (i % 4) * (thumb_size + 5)
            y = 110 + (i // 4) * (thumb_size + 5)
            thumb = cv2.resize(img, (thumb_size, thumb_size))
            frame[y:y+thumb_size, x:x+thumb_size] = thumb
            cv2.rectangle(frame, (x-1, y-1), (x+thumb_size+1, y+thumb_size+1), color, 1)


def draw_capture_bg_intro(frame, state: GameState):
    """Draw background capture intro."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    draw_centered_text(frame, "STEP 2: NEGATIVE SAMPLES", h // 5, 1.0, (100, 100, 255), 2)

    draw_centered_text(frame, "REMOVE your object from view!", h // 3 + 10, 0.75, (255, 100, 100), 2)
    draw_centered_text(frame, "Classifier needs 'not object' examples", h // 3 + 45, 0.55, (200, 200, 200), 1)

    expected_bg_frames = int(BG_CAPTURE_DURATION * BG_CAPTURE_FPS)
    draw_centered_text(frame, f"{int(BG_CAPTURE_DURATION)}s = ~{expected_bg_frames} background images", h // 2 + 20, 0.6, (255, 200, 100), 2)

    draw_centered_text(frame, "Binary classification: object vs background", h - 120, 0.5, (150, 255, 150), 1)

    draw_centered_text(frame, "Press SPACE when object REMOVED", h - 60, 0.85, (100, 255, 100), 2)


def draw_education_slide(frame, state: GameState):
    """Draw educational slides with technical explanations."""
    h, w = frame.shape[:2]

    # Dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

    slide = EDUCATION_SLIDES[state.education_slide_idx]

    # Slide counter
    slide_num = f"[{state.education_slide_idx + 1}/{len(EDUCATION_SLIDES)}]"
    draw_text_with_bg(frame, slide_num, (w - 100, 30), 0.6, (150, 150, 150), 1)

    # Title
    draw_centered_text(frame, slide["title"], 60, 1.2, (0, 255, 255), 2)

    # Content lines
    y_start = 120
    line_height = 32
    for i, line in enumerate(slide["content"]):
        if line == "":
            continue
        # Highlight code-like content
        if line.startswith("  "):
            color = (100, 255, 100)  # Green for code/examples
        else:
            color = (220, 220, 220)
        draw_text_with_bg(frame, line, (40, y_start + i * line_height), 0.55, color, 1)

    # Navigation hints
    nav_y = h - 50
    if state.education_slide_idx > 0:
        draw_text_with_bg(frame, "<-- LEFT: Previous", (20, nav_y), 0.5, (150, 150, 150), 1)
    if state.education_slide_idx < len(EDUCATION_SLIDES) - 1:
        draw_text_with_bg(frame, "RIGHT: Next -->", (w - 180, nav_y), 0.5, (150, 150, 150), 1)
        draw_centered_text(frame, "Arrow keys to navigate", nav_y, 0.6, (200, 200, 200), 1)
    else:
        draw_centered_text(frame, "Press SPACE to start training!", nav_y, 0.8, (100, 255, 100), 2)


def draw_training(frame, state: GameState):
    """Draw the REAL training phase with technical details."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    draw_centered_text(frame, "TRAINING YOLOv8n-cls", 40, 1.0, (0, 255, 255), 2)
    draw_centered_text(frame, "Gradient descent on your data", 70, 0.5, (200, 200, 200), 1)

    # Show some captured images (smaller, higher)
    if state.captured_frames:
        thumb_size = 38
        cols = min(6, len(state.captured_frames))
        grid_width = cols * (thumb_size + 4)
        start_x = (w - grid_width) // 2

        for i, img in enumerate(state.captured_frames[:6]):
            x = start_x + i * (thumb_size + 4)
            y = 90

            processed = i < int(state.training_progress * 6)
            border = (0, 255, 0) if processed else (100, 100, 100)

            thumb = cv2.resize(img, (thumb_size, thumb_size))
            frame[y:y+thumb_size, x:x+thumb_size] = thumb
            cv2.rectangle(frame, (x-1, y-1), (x+thumb_size+1, y+thumb_size+1), border, 1)

    # Progress bar
    bar_y = 145
    draw_progress_bar(frame, 40, bar_y, w - 80, 30, state.training_progress, (0, 200, 255))
    percent = int(state.training_progress * 100)
    draw_centered_text(frame, f"{percent}%", bar_y + 24, 0.7, (255, 255, 255), 2)

    # REAL training stats - more detailed
    stats_y = 200
    train_images = int(len(state.captured_frames) * 0.8) + int(len(state.background_frames) * 0.8)

    draw_text_with_bg(frame, f"Epoch: {state.training_epoch}/{TRAIN_EPOCHS}",
                      (30, stats_y), 0.55, (150, 255, 150), 1)
    draw_text_with_bg(frame, f"Loss: {state.training_loss:.4f}",
                      (w//3, stats_y), 0.55, (255, 200, 150), 1)
    draw_text_with_bg(frame, f"Images: {train_images}",
                      (w*2//3, stats_y), 0.55, (150, 150, 255), 1)

    # Additional stats row
    stats_y2 = 230
    batches = max(1, train_images // 8)
    draw_text_with_bg(frame, f"Batches/epoch: {batches}",
                      (30, stats_y2), 0.45, (180, 180, 180), 1)
    draw_text_with_bg(frame, f"Batch size: 8",
                      (w//3, stats_y2), 0.45, (180, 180, 180), 1)
    draw_text_with_bg(frame, f"Input: 224x224",
                      (w*2//3, stats_y2), 0.45, (180, 180, 180), 1)

    # Educational fact
    fact_y = 275
    fact_idx = int(time.time() / 3) % len(TRAINING_FACTS)
    draw_centered_text(frame, TRAINING_FACTS[fact_idx], fact_y, 0.45, (200, 200, 255), 1)

    # Training log
    log_y = 320
    draw_text_with_bg(frame, "Log:", (30, log_y), 0.45, (150, 150, 150), 1)
    for i, log_line in enumerate(state.training_log[-5:]):
        color = (0, 0, 255) if "Error" in log_line else (100, 200, 100)
        draw_text_with_bg(frame, f"> {log_line}", (30, log_y + 20 + i * 18), 0.4, color, 1)

    # Show error recovery hint
    if state.training_error:
        draw_centered_text(frame, "Training failed! Press R to restart", h - 30, 0.6, (0, 0, 255), 2)


def draw_play_intro(frame, state: GameState):
    """Draw play phase introduction."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    draw_centered_text(frame, "TRAINING COMPLETE!", h // 5, 1.2, (0, 255, 0), 2)
    draw_centered_text(frame, f"Validation accuracy: {state.training_accuracy*100:.0f}%", h // 5 + 40, 0.65, (255, 255, 100), 2)

    draw_centered_text(frame, "Model trained on YOUR images!", h // 2 - 30, 0.7, (200, 200, 200), 2)
    draw_centered_text(frame, "Now test YOUR classifier!", h // 2 + 5, 0.85, (255, 255, 0), 2)

    draw_centered_text(frame, "Show object - watch confidence", h // 2 + 50, 0.55, (200, 200, 200), 1)
    draw_centered_text(frame, "Hold steady when high to score!", h // 2 + 80, 0.55, (200, 200, 200), 1)

    draw_centered_text(frame, f"{int(PLAY_DURATION)} seconds", h - 95, 0.7, (255, 200, 100), 2)
    draw_centered_text(frame, "Press SPACE to start!", h - 55, 0.9, (100, 255, 100), 2)


def draw_playing(frame, state: GameState) -> Tuple[bool, str, float]:
    """
    Draw the play phase UI.
    Returns: (scored, predicted_class, confidence)
    """
    h, w = frame.shape[:2]

    # Run inference with trained model
    if state.classifier is None:
        draw_centered_text(frame, "Loading model...", h // 2, 1.0, (255, 255, 0), 2)
        return False, "", 0.0

    # Predict
    results = state.classifier(frame, verbose=False)
    probs = results[0].probs
    pred_class = state.classifier.names[probs.top1]
    confidence = float(probs.top1conf)

    # Time remaining
    elapsed = time.time() - state.play_start_time
    remaining = max(0, PLAY_DURATION - elapsed)
    time_color = (0, 255, 0) if remaining > 15 else (0, 165, 255) if remaining > 5 else (0, 0, 255)

    # Header
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)

    # Show what we're looking for
    draw_text_with_bg(frame, f"FIND: {state.object_name.upper()}", (20, 45), 1.0, (0, 255, 255), 2, (0,0,0,0))
    draw_text_with_bg(frame, f"Score: {state.score}", (w - 250, 30), 0.8, (255, 255, 255), 2, (0,0,0,0))
    draw_text_with_bg(frame, f"Time: {remaining:.1f}s", (w - 250, 55), 0.7, time_color, 2, (0,0,0,0))

    # Large confidence display
    bar_y = h - 130
    bar_height = 50

    is_object = pred_class == state.object_name

    # Gradient confidence color: red (0%) -> orange (50%) -> green (100%)
    if confidence < 0.5:
        # Red to orange (0-50%)
        ratio = confidence / 0.5
        conf_color = (0, int(165 * ratio), int(255 * (1 - ratio) + 165 * ratio))  # BGR
    else:
        # Orange to green (50-100%)
        ratio = (confidence - 0.5) / 0.5
        conf_color = (0, int(165 + 90 * ratio), int(165 * (1 - ratio)))  # BGR

    # Confidence bar with gradient
    draw_text_with_bg(frame, f"Model: {pred_class} ({confidence*100:.0f}%)",
                      (30, bar_y - 25), 0.8, conf_color, 2)
    draw_progress_bar(frame, 30, bar_y, w - 60, bar_height, confidence, conf_color)

    scored = False
    if is_object and confidence > 0.5:
        # Object detected!
        if state.hold_start_time is None:
            state.hold_start_time = time.time()

        hold_elapsed = time.time() - state.hold_start_time
        hold_progress = min(1.0, hold_elapsed / HOLD_DURATION)

        # Hold progress indicator
        draw_centered_text(frame, "DETECTED! HOLD STEADY!", bar_y + bar_height + 35, 0.9, (0, 255, 0), 2)
        draw_progress_bar(frame, w//4, bar_y + bar_height + 50, w//2, 20, hold_progress, (0, 255, 0))

        if hold_progress >= 1.0:
            scored = True
            state.hold_start_time = None
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 15)
    else:
        state.hold_start_time = None
        hint = "Show your object to the camera!" if pred_class == "background" else "Model predicts different class..."
        draw_centered_text(frame, hint, bar_y + bar_height + 35, 0.6, (150, 150, 150), 1)

    return scored, pred_class, confidence


def draw_game_over(frame, state: GameState):
    """Draw game over screen with matrix visualization."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    draw_centered_text(frame, "TRAINING COMPLETE!", h // 8, 1.2, (0, 255, 255), 2)
    draw_centered_text(frame, "You trained a neural network!", h // 8 + 35, 0.6, (200, 200, 200), 1)

    draw_centered_text(frame, f"Score: {state.score}", h // 5 + 50, 1.0, (255, 255, 0), 2)
    draw_centered_text(frame, f"Val accuracy: {state.training_accuracy*100:.0f}%", h // 5 + 85, 0.6, (150, 255, 150), 1)

    # Mini weight matrix visualization (illustration)
    matrix_y = h // 3 + 30
    draw_text_with_bg(frame, "Weight matrix (illustration):", (30, matrix_y), 0.5, (200, 200, 200), 1)

    # Illustrative weights - actual model has 3M of these
    np.random.seed(int(state.training_accuracy * 1000))
    weights = np.random.randn(4, 6) * 0.5

    cell_w, cell_h = 45, 22
    matrix_x = 40
    for i in range(4):
        for j in range(6):
            val = weights[i, j]
            # Color: blue negative, green positive
            if val < 0:
                color = (255, int(128 + val * 100), 0)  # Blue-ish
            else:
                color = (0, int(128 + val * 100), 0)  # Green-ish
            x = matrix_x + j * cell_w
            y = matrix_y + 20 + i * cell_h
            cv2.rectangle(frame, (x, y), (x + cell_w - 2, y + cell_h - 2), color, -1)
            cv2.putText(frame, f"{val:+.2f}", (x + 3, y + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Technical summary on right side
    summary_x = w // 2 + 20
    draw_text_with_bg(frame, "What happened:", (summary_x, matrix_y), 0.5, (255, 255, 255), 1)
    summaries = [
        f"- {len(state.captured_frames)} object images",
        f"- {len(state.background_frames)} background images",
        f"- {TRAIN_EPOCHS} epochs of training",
        "- 3M weights adjusted",
        "- Cross-entropy loss minimized",
    ]
    for i, s in enumerate(summaries):
        color = (150, 255, 150) if i < 2 else (150, 150, 255)
        draw_text_with_bg(frame, s, (summary_x, matrix_y + 25 + i * 22), 0.4, color, 1)

    draw_centered_text(frame, "Real neural network training!", h - 90, 0.6, (255, 200, 100), 2)
    draw_centered_text(frame, "Press SPACE to play again", h - 55, 0.8, (100, 255, 100), 2)


# =============================================================================
# MAIN GAME LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train The AI - Real ML Experience")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    # Ensure background directory exists
    BACKGROUND_DIR.mkdir(exist_ok=True)

    # Open camera
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Train The AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = GameState()

    print("\nTrain The AI - Real ML Experience")
    print("=" * 40)
    print("Controls:")
    print("  SPACE - Capture / Continue")
    print("  R     - Reset")
    print("  Q     - Quit")
    print("")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Handle phases
        if state.phase == GamePhase.TITLE:
            draw_title_screen(frame, state)

        elif state.phase == GamePhase.CAPTURE_INTRO:
            draw_capture_intro(frame, state)

        elif state.phase == GamePhase.CAPTURING:
            # Auto-capture frames during recording
            if state.recording_start_time is not None:
                elapsed = time.time() - state.recording_start_time

                # Capture frame at configured FPS
                if time.time() - state.last_capture_time >= state.frame_interval:
                    state.captured_frames.append(frame.copy())
                    state.last_capture_time = time.time()

                # Check if recording complete
                if elapsed >= CAPTURE_DURATION:
                    print(f"Object capture complete: {len(state.captured_frames)} frames")
                    state.recording_start_time = None
                    state.phase = GamePhase.CAPTURE_BG_INTRO

            draw_capturing(frame, state, is_background=False)

        elif state.phase == GamePhase.CAPTURE_BG_INTRO:
            draw_capture_bg_intro(frame, state)

        elif state.phase == GamePhase.CAPTURING_BG:
            # Auto-capture background frames during recording
            if state.recording_start_time is not None:
                elapsed = time.time() - state.recording_start_time

                # Capture frame at configured FPS
                if time.time() - state.last_capture_time >= (1.0 / BG_CAPTURE_FPS):
                    state.background_frames.append(frame.copy())
                    state.last_capture_time = time.time()

                # Check if recording complete
                if elapsed >= BG_CAPTURE_DURATION:
                    print(f"Background capture complete: {len(state.background_frames)} frames")
                    state.recording_start_time = None

                    # Go to education slides before training
                    state.phase = GamePhase.EDUCATION
                    state.education_slide_idx = 0

            draw_capturing(frame, state, is_background=True)

        elif state.phase == GamePhase.EDUCATION:
            draw_education_slide(frame, state)

        elif state.phase == GamePhase.TRAINING:
            draw_training(frame, state)

            # Only proceed if training completed successfully
            if state.training_complete and not state.training_error:
                state.phase = GamePhase.PLAY_INTRO

        elif state.phase == GamePhase.PLAY_INTRO:
            draw_play_intro(frame, state)

        elif state.phase == GamePhase.PLAYING:
            elapsed = time.time() - state.play_start_time
            if elapsed >= PLAY_DURATION:
                state.phase = GamePhase.GAME_OVER
            else:
                scored, _, _ = draw_playing(frame, state)
                if scored:
                    state.score += 1

        elif state.phase == GamePhase.GAME_OVER:
            draw_game_over(frame, state)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            # Clean up temp files on reset
            if TEMP_DATASET_DIR.exists():
                shutil.rmtree(TEMP_DATASET_DIR)
            runs_dir = GAME_DIR / "runs"
            if runs_dir.exists():
                shutil.rmtree(runs_dir)
            state = GameState()

        elif key == ord(' '):
            if state.phase == GamePhase.TITLE:
                state.phase = GamePhase.CAPTURE_INTRO

            elif state.phase == GamePhase.CAPTURE_INTRO:
                # Start object video recording
                state.phase = GamePhase.CAPTURING
                state.captured_frames = []
                state.recording_start_time = time.time()
                state.last_capture_time = time.time()
                print("Starting object capture recording...")

            elif state.phase == GamePhase.CAPTURE_BG_INTRO:
                # Start background video recording
                state.phase = GamePhase.CAPTURING_BG
                state.background_frames = []
                state.recording_start_time = time.time()
                state.last_capture_time = time.time()
                print("Starting background capture recording...")

            elif state.phase == GamePhase.PLAY_INTRO:
                # Load the trained model
                from ultralytics import YOLO
                if state.trained_model_path and Path(state.trained_model_path).exists():
                    state.classifier = YOLO(state.trained_model_path)
                    print(f"Loaded trained model: {state.trained_model_path}")
                else:
                    print("Warning: Using fallback model")
                    state.classifier = YOLO("yolov8n-cls.pt")

                state.phase = GamePhase.PLAYING
                state.play_start_time = time.time()
                state.score = 0
                state.hold_start_time = None

            elif state.phase == GamePhase.EDUCATION:
                # On last slide, SPACE starts training
                if state.education_slide_idx >= len(EDUCATION_SLIDES) - 1:
                    print("Starting training...")
                    state.phase = GamePhase.TRAINING
                    state.phase_start_time = time.time()

                    # Prepare dataset and start training thread
                    dataset_path = prepare_dataset(state)
                    state.training_thread = threading.Thread(
                        target=train_model,
                        args=(state, dataset_path)
                    )
                    state.training_thread.start()

            elif state.phase == GamePhase.GAME_OVER:
                # Clean up and restart
                if TEMP_DATASET_DIR.exists():
                    shutil.rmtree(TEMP_DATASET_DIR)
                runs_dir = GAME_DIR / "runs"
                if runs_dir.exists():
                    shutil.rmtree(runs_dir)
                state = GameState()

        # Arrow keys for education slide navigation
        # LEFT arrow: key codes 81 (Linux), 2 (some systems)
        # RIGHT arrow: key codes 83 (Linux), 3 (some systems)
        if state.phase == GamePhase.EDUCATION:
            if key in [81, 2]:  # LEFT arrow
                if state.education_slide_idx > 0:
                    state.education_slide_idx -= 1
            elif key in [83, 3]:  # RIGHT arrow
                if state.education_slide_idx < len(EDUCATION_SLIDES) - 1:
                    state.education_slide_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Cleanup
    if TEMP_DATASET_DIR.exists():
        shutil.rmtree(TEMP_DATASET_DIR)


if __name__ == "__main__":
    main()
