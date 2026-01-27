# MITCH 2026 - Modified for ML Visions Launcher

# System Modules
from pathlib import Path
from enum import Enum
import random
import time
import threading
import subprocess
import sys

# Installed Modules
import cv2
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Project Modules - use relative import for config in same package
from games.TileShuffle.config import ASSETS_DIR, DISPLAY_RES, ANIM_RES, TILE_SIZE
from inference.onnx_infer import YOLOClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model path
MODEL_PATH = PROJECT_ROOT / "models" / "7class_v1" / "best_xavier.onnx"

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection settings
INFERENCE_INTERVAL = 0.1  # seconds between inferences
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence to act on detection

# Timing settings
MENU_TIMEOUT = 5.0  # seconds of no detection before returning to idle
CONFIRM_DURATION = 2.0  # seconds to hold object before launching

# Object to game mapping
GAME_MAPPINGS = {
    "hammer": {
        "name": "Demon Quest",
        "description": "Battle demons with real objects!",
        "command": ["python3", str(PROJECT_ROOT / "games" / "DemonQuest" / "quest_projection.py")],
    },
    "9v_battery": {
        "name": "Train Classifier",
        "description": "Train your own AI model!",
        "command": ["python3", str(PROJECT_ROOT / "games" / "TrainClassifier" / "game.py")],
    },
    "blue_floppy": {
        "name": "Scavenger Hunt",
        "description": "Find objects around the space!",
        "command": ["python3", str(PROJECT_ROOT / "games" / "SimpleHunt" / "game.py")],
    },
}

# Wake gesture
WAKE_OBJECT = "hand"


# =============================================================================
# STATE MACHINE
# =============================================================================

class State(Enum):
    IDLE = "idle"
    MENU = "menu"
    CONFIRMING = "confirming"
    LAUNCHING = "launching"


# =============================================================================
# CAMERA THREAD
# =============================================================================

class CameraThread:
    """Background thread for camera capture."""

    def __init__(self, camera_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if not self.cap.isOpened():
            print(f"Warning: Could not open camera {CAMERA_INDEX}")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Mirror for intuitive interaction
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
            time.sleep(0.016)  # ~60fps capture

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


# =============================================================================
# TILE ANIMATION (Original Code)
# =============================================================================

# --- Load all tile images ---
tile_dir = ASSETS_DIR / "tiles"
tile_paths = list(tile_dir.glob("*.*"))
if not tile_paths:
    raise FileNotFoundError("No tile images found in folder.")

noisy_background = cv2.imread(str(ASSETS_DIR / "noisy_canvas.png"))


def load_random_tile(W, H):
    path = random.choice(tile_paths)
    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = (0, 255, 0)
    else:
        img = cv2.resize(img, (W, H))
    return img


def generate_display_bg():
    # --- Generate full display background with random tiles ---
    display_background = np.zeros((DISPLAY_RES[1], DISPLAY_RES[0], 3), dtype=np.uint8)
    for y in range(0, DISPLAY_RES[1], TILE_SIZE):
        for x in range(0, DISPLAY_RES[0], TILE_SIZE):
            tile_img = load_random_tile(TILE_SIZE, TILE_SIZE)
            y1, y2 = y, min(y + TILE_SIZE, DISPLAY_RES[1])
            x1, x2 = x, min(x + TILE_SIZE, DISPLAY_RES[0])
            display_background[y1:y2, x1:x2] = tile_img[:y2-y1, :x2-x1]
    return display_background


class ImageTile:
    def __init__(self, L, T, W, H, image_dir=tile_dir):
        self.L = L
        self.T = T
        self.W = W
        self.H = H

        # Load random image if directory provided
        self.image = np.zeros((H, W, 3), dtype=np.uint8)
        self.image[:, :] = (0, 255, 0)  # default green

        if image_dir:
            images = list(Path(image_dir).glob("*.*"))
            if images:
                img_path = random.choice(images)
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.image = cv2.resize(img, (W, H))

        self.base_image = self.image.copy()
        self.clear_letter()
        self.cooldown = 0

    def on_moved(self, cooldown=1):
        self.cooldown = cooldown

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def clear_letter(self):
        self.image = self.base_image.copy()
        self.current_letter = None

    def set_letter(self, letter):
        self.clear_letter()
        self.current_letter = letter

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 4

        while True:
            (w, h), _ = cv2.getTextSize(letter, font, scale, thickness)
            if w < self.W * 0.8 and h < self.H * 0.8:
                scale += 0.1
            else:
                scale -= 0.1
                break

        (w, h), _ = cv2.getTextSize(letter, font, scale, thickness)
        x = (self.W - w) // 2
        y = (self.H + h) // 2

        cv2.putText(
            self.image, letter, (x, y),
            font, scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )

    def draw(self, canvas, x=None, y=None):
        L = self.L if x is None else x
        T = self.T if y is None else y
        canvas[T:T+self.H, L:L+self.W] = self.image


class SlidingTileAnimation:
    def __init__(self, grid_size=16, canvas_size=1024, slide_steps=45):
        self.W = canvas_size
        self.H = canvas_size
        self.GRID_SIZE = grid_size
        self.TILE_SIZE = TILE_SIZE
        self.slide_steps = slide_steps

        # Canvas for animation (1024x1024)
        self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Create tiles
        self.tiles = []
        letters = 25 * ("                " + "  denhac rules  " + "                ")
        for y in range(0, self.H, self.TILE_SIZE):
            for x in range(0, self.W, self.TILE_SIZE):
                self.tiles.append(ImageTile(x, y, self.TILE_SIZE, self.TILE_SIZE))
                self.tiles[-1].set_letter(letters[len(self.tiles)-1])

        self.redraw_middle_rows()

        # Pick random empty tile
        tile = random.choice(self.tiles)
        self.empty_pos = (tile.L, tile.T)
        self.tiles.remove(tile)
        self.last_moved = None

        # Sliding state
        self.moving_tile = None
        self.start_pos = None
        self.end_pos = None
        self.current_slide = 0

    def tiles_in_row(self, row_index):
        y = row_index * self.TILE_SIZE
        row_tiles = [t for t in self.tiles if t.T == y]
        return sorted(row_tiles, key=lambda t: t.L)

    def redraw_middle_rows(self):
        for row_index in range(0, 16):
            row_tiles = self.tiles_in_row(row_index)
            if row_index in [1, 4, 7, 10, 13]:
                message = " " * random.randint(1, 3)
                message += "denhac rules"
                message = message.ljust(16)

                for tile, char in zip(row_tiles, message.ljust(len(row_tiles))):
                    if char == " ":
                        tile.clear_letter()
                    else:
                        tile.set_letter(char)
            else:
                for tile in row_tiles:
                    tile.clear_letter()

    def get_neighbors(self, pos):
        x, y = pos
        delta = self.TILE_SIZE

        candidates = []
        for dx, dy in [(-delta, 0), (delta, 0), (0, -delta), (0, delta)]:
            nx, ny = x + dx, y + dy
            for tile in self.tiles:
                if (tile.L, tile.T) == (nx, ny):
                    candidates.append(tile)

        if not candidates:
            return []

        # Try selecting a tile, respecting cooldown probabilistically
        for _ in range(3):
            tile = random.choice(candidates)
            if tile.cooldown <= 0:
                return [tile]
            else:
                tile.cooldown -= 1

        return [random.choice(candidates)]

    def read(self):
        self.canvas[:] = noisy_background
        # Start new slide if none active
        if self.moving_tile is None:
            neighbors = self.get_neighbors(self.empty_pos)
            if neighbors:
                self.moving_tile = random.choice(neighbors)
                self.start_pos = (self.moving_tile.L, self.moving_tile.T)
                self.end_pos = self.empty_pos
                self.current_slide = 0

                self.slide_steps += random.randint(-2, 5)
                if self.slide_steps <= 15:
                    self.slide_steps = 15
                elif self.slide_steps >= 30:
                    self.slide_steps = 30

        # Draw static tiles
        for tile in self.tiles:
            if tile != self.moving_tile:
                tile.draw(self.canvas)
            tile.tick_cooldown()

        # Animate moving tile
        if self.moving_tile:
            t = self.current_slide / self.slide_steps
            cur_x = int(self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t)
            cur_y = int(self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t)
            self.moving_tile.draw(self.canvas, cur_x, cur_y)
            self.current_slide += 1
            if self.current_slide > self.slide_steps:
                self.moving_tile.L, self.moving_tile.T = self.end_pos
                self.moving_tile.on_moved(cooldown=5)

                if self.moving_tile.current_letter is None:
                    self.slide_steps -= 30

                self.last_moved = self.moving_tile
                self.empty_pos = self.start_pos
                self.moving_tile = None

        return self.canvas


# =============================================================================
# MENU UI DRAWING
# =============================================================================

def draw_text_centered(img, text, y, font_scale=1.0, color=(255, 255, 255), thickness=2):
    """Draw text centered horizontally at given y position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_idle_overlay(canvas):
    """Draw 'Raise your hand to play' prompt on idle screen."""
    h, w = canvas.shape[:2]

    # Semi-transparent bar at bottom
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    # Text
    draw_text_centered(canvas, "Raise your hand to play", h - 35, 1.0, (255, 255, 255), 2)


def draw_menu_overlay(canvas, camera_frame, detected_class=None, confirm_progress=0.0):
    """Draw the full menu overlay with camera feed and game options."""
    h, w = canvas.shape[:2]

    # Dim the background
    canvas[:] = (canvas * 0.3).astype(np.uint8)

    # Draw menu box
    margin = 60
    box_color = (80, 80, 80)
    border_color = (200, 200, 200)

    cv2.rectangle(canvas, (margin, margin), (w - margin, h - margin), box_color, -1)
    cv2.rectangle(canvas, (margin, margin), (w - margin, h - margin), border_color, 3)

    # Title
    draw_text_centered(canvas, "WELCOME TO ML VISIONS", margin + 50, 1.2, (255, 255, 255), 2)

    # Instructions
    draw_text_centered(canvas, "Pick a game by showing an object to the camera:", margin + 100, 0.7, (200, 200, 200), 1)

    # Game options
    y_start = margin + 160
    line_height = 50
    games_info = [
        ("Hammer", "Demon Quest", "hammer"),
        ("9V Battery", "Train Classifier", "9v_battery"),
        ("Floppy Disk", "Scavenger Hunt", "blue_floppy"),
    ]

    for i, (obj, game, class_name) in enumerate(games_info):
        y = y_start + i * line_height
        text = f"{obj} ................... {game}"

        # Highlight if this game is selected
        if detected_class == class_name:
            color = (0, 255, 255)  # Yellow highlight
            thickness = 2
        else:
            color = (255, 255, 255)
            thickness = 1

        draw_text_centered(canvas, text, y, 0.7, color, thickness)

    # Camera feed box
    cam_w, cam_h = 240, 180
    cam_x = (w - cam_w) // 2
    cam_y = y_start + len(games_info) * line_height + 30

    # Draw camera frame
    if camera_frame is not None:
        cam_resized = cv2.resize(camera_frame, (cam_w, cam_h))
        canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = cam_resized

    # Camera border
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), border_color, 2)

    # Confirmation progress bar (if confirming)
    if confirm_progress > 0:
        bar_y = cam_y + cam_h + 20
        bar_w = 300
        bar_h = 20
        bar_x = (w - bar_w) // 2

        # Background
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Progress
        progress_w = int(bar_w * confirm_progress)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), (0, 255, 0), -1)
        # Border
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), border_color, 1)

        # Text above progress bar
        if detected_class and detected_class in GAME_MAPPINGS:
            game_name = GAME_MAPPINGS[detected_class]["name"]
            draw_text_centered(canvas, f"Hold to launch {game_name}...", bar_y - 10, 0.6, (0, 255, 255), 1)

    # Divider line
    div_y = cam_y + cam_h + 70
    cv2.line(canvas, (margin + 40, div_y), (w - margin - 40, div_y), border_color, 1)

    # Community info section
    info_y = div_y + 30
    info_lines = [
        ("Interested? Come join us!", 0.7, (255, 255, 255)),
        ("", 0.5, (200, 200, 200)),
        ("SCHEDULE", 0.6, (200, 200, 200)),
        ("1st & 3rd Mondays, 6-8pm", 0.5, (180, 180, 180)),
        ("Sunday between those Mondays, 9-11am", 0.5, (180, 180, 180)),
        ("", 0.5, (200, 200, 200)),
        ("CONNECT", 0.6, (200, 200, 200)),
        ("Slack: #ml-visions", 0.5, (180, 180, 180)),
        ("Also check out: #ai-compsci-club", 0.5, (180, 180, 180)),
        ("", 0.5, (200, 200, 200)),
        ("Built by the ML Visions group at denhac", 0.5, (150, 150, 150)),
    ]

    for i, (text, scale, color) in enumerate(info_lines):
        if text:
            draw_text_centered(canvas, text, info_y + i * 28, scale, color, 1)

    # Footer
    draw_text_centered(canvas, "(Step away to return)", h - margin - 20, 0.5, (150, 150, 150), 1)


# =============================================================================
# MAIN LAUNCHER
# =============================================================================

class Launcher:
    def __init__(self):
        self.state = State.IDLE
        self.anim = SlidingTileAnimation()

        # Camera and inference
        self.camera = CameraThread(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
        self.classifier = None

        # Timing
        self.last_inference_time = 0
        self.last_detection_time = 0
        self.confirm_start_time = 0
        self.background_time = 0

        # Current detection
        self.current_detection = None
        self.current_confidence = 0.0
        self.selected_game = None

        # Display backgrounds
        self.display_bgs = [generate_display_bg() for _ in range(10)]
        self.bg_index = 0
        self.anim_reset_time = time.time()

    def initialize(self):
        """Initialize camera and model."""
        print("Initializing ML Visions Launcher...")

        # Start camera
        if not self.camera.start():
            print("Warning: Running without camera")

        # Load model - try TensorRT first (best for Xavier), then CUDA, then CPU
        if MODEL_PATH.exists():
            print(f"Loading model: {MODEL_PATH}")
            # TensorRT is optimized for Xavier and shouldn't deadlock like raw CUDA
            self.classifier = YOLOClassifier(
                str(MODEL_PATH),
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print(f"Using providers: {self.classifier.session.get_providers()}")
            print(f"Classes: {self.classifier.classes}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            print("Running without object detection")

    def run_inference(self):
        """Run inference on current camera frame."""
        if self.classifier is None:
            return None, 0.0

        frame = self.camera.get_frame()
        if frame is None:
            return None, 0.0

        pred_class, confidence = self.classifier.predict(frame)
        return pred_class, confidence

    def update_state(self):
        """Update state machine based on detections."""
        now = time.time()

        # Run inference periodically
        if now - self.last_inference_time >= INFERENCE_INTERVAL:
            self.last_inference_time = now
            detected, confidence = self.run_inference()

            if detected and confidence >= CONFIDENCE_THRESHOLD:
                if detected=="background" and detected!=self.current_detection:
                    self.background_time = now
                self.current_detection = detected
                self.current_confidence = confidence
                self.last_detection_time = now
            else:
                self.current_detection = None
                self.current_confidence = 0.0

        # State transitions
        if self.state == State.IDLE:
            # Wake up on hand detection
            if self.current_detection == WAKE_OBJECT:
                print("Hand detected - opening menu")
                self.state = State.MENU
                self.last_detection_time = now

        elif self.state == State.MENU:
            # Check for game selection
            
            if self.current_detection in GAME_MAPPINGS:
                self.selected_game = self.current_detection
                self.confirm_start_time = now
                self.state = State.CONFIRMING
                print(f"Selecting: {GAME_MAPPINGS[self.selected_game]['name']}")

            # Timeout back to idle
            elif self.current_detection=="background":
                if now-self.background_time>= MENU_TIMEOUT:
                    print("Menu timeout - returning to idle")
                    self.state = State.IDLE
                    self.selected_game = None

        elif self.state == State.CONFIRMING:
            # Check if still holding the same object
            if self.current_detection == self.selected_game:
                # Check if held long enough
                if now - self.confirm_start_time >= CONFIRM_DURATION:
                    self.state = State.LAUNCHING
            else:
                # Released or changed object - back to menu
                print("Selection cancelled")
                self.state = State.MENU
                self.selected_game = None
                self.confirm_start_time = 0

        elif self.state == State.LAUNCHING:
            self.launch_game()
            self.state = State.IDLE
            self.selected_game = None

    def launch_game(self):
        """Launch the selected game."""
        if self.selected_game not in GAME_MAPPINGS:
            return

        game = GAME_MAPPINGS[self.selected_game]
        print(f"Launching: {game['name']}")

        # Release camera so game can use it
        print("Releasing camera for game...")
        self.camera.stop()

        try:
            # Run game as subprocess (blocks until game exits)
            subprocess.run(game["command"], cwd=str(PROJECT_ROOT))
        except Exception as e:
            print(f"Error launching game: {e}")

        print("Game exited - restarting camera...")
        # Restart camera after game exits
        self.camera = CameraThread(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
        self.camera.start()

        print("Returning to idle")

    def render(self):
        """Render current frame based on state."""
        # Get animation frame
        anim_frame = self.anim.read()

        # Check for animation reset
        if time.time() - self.anim_reset_time >= 300:
            self.anim.redraw_middle_rows()
            self.bg_index = (self.bg_index + 1) % len(self.display_bgs)
            self.anim_reset_time = time.time()

        # Compose onto display background
        display_bg = self.display_bgs[self.bg_index]
        final_canvas = display_bg.copy()
        start_x = (DISPLAY_RES[0] - ANIM_RES[0]) // 2
        start_y = (DISPLAY_RES[1] - ANIM_RES[1]) // 2
        final_canvas[start_y:start_y + ANIM_RES[1], start_x:start_x + ANIM_RES[0]] = anim_frame

        # Draw overlays based on state
        if self.state == State.IDLE:
            draw_idle_overlay(final_canvas)

        elif self.state in (State.MENU, State.CONFIRMING):
            camera_frame = self.camera.get_frame()

            # Calculate confirm progress
            confirm_progress = 0.0
            if self.state == State.CONFIRMING and self.confirm_start_time > 0:
                elapsed = time.time() - self.confirm_start_time
                confirm_progress = min(elapsed / CONFIRM_DURATION, 1.0)

            draw_menu_overlay(
                final_canvas,
                camera_frame,
                detected_class=self.selected_game if self.state == State.CONFIRMING else self.current_detection,
                confirm_progress=confirm_progress
            )

        return final_canvas

    def run(self):
        """Main loop."""
        self.initialize()

        delay = int(1000 / 59.94)
        cv2.namedWindow("SlidingTiles", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("SlidingTiles", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("Launcher running. Press 'q' to quit.")

        try:
            while True:
                # Update state machine
                self.update_state()

                # Render frame
                frame = self.render()

                # Display
                cv2.imshow("SlidingTiles", frame)

                # Handle input
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                # Debug: 'm' to toggle menu
                elif key == ord('m'):
                    if self.state == State.IDLE:
                        self.state = State.MENU
                        self.last_detection_time = time.time()
                    else:
                        self.state = State.IDLE

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            print("Launcher stopped.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    launcher = Launcher()
    launcher.run()
