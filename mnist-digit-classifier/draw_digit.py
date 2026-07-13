import tkinter as tk
from tkinter import font
from PIL import Image, ImageDraw
import numpy as np

# ── Import network functions from your training file ──
from digit_recognizer import forward, softmax, relu

# ── Load trained params ──
# params.npy is saved in ml-from-scratch root, one level up from mnist-digit-classifier
params   = np.load('mnist-digit-classifier/params.npy', allow_pickle=True).item()
n_layers = len(params) // 2  # number of weight layers (W1,B1,W2,B2... -> divide by 2)

CANVAS_SIZE = 280  # physical canvas size in pixels (10x zoom of 28x28)
DIGIT_SIZE  = 28   # what the network expects
BRUSH_SIZE  = 10   # stroke thickness when drawing


def preprocess_canvas(pil_image):
    # Scale 280x280 PIL image down to 28x28 - each 10x10 block averages into one pixel
    small = pil_image.resize((DIGIT_SIZE, DIGIT_SIZE), Image.LANCZOS)

    # Convert to numpy array, shape (28,28), values 0-255
    arr = np.array(small, dtype=np.float32)

    # Normalise to 0-1 to match training preprocessing
    arr = arr / 255.0

    # Flatten to (784, 1) column vector - shape the network input layer expects
    arr = arr.reshape(784, 1)

    return arr


def predict(arr):
    # Run the input through the trained network
    cached = forward(arr, params, n_layers)
    output = cached[f'A{n_layers}']  # shape (10, 1) - probability per digit
    digit  = int(np.argmax(output))   # index of highest probability = predicted digit
    conf   = float(output[digit][0]) * 100 # softmax value as percentage confidence
    return digit, conf


class DrawApp:
    def __init__(self, root):
        self.root = root
        root.title("Digit Recogniser")
        root.resizable(False, False)
        root.configure(bg="#1a1a1a")

        # PIL image mirrors what is drawn on screen - used for prediction
        # "L" = greyscale mode, colour 0 = black background
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        pad = dict(padx=20, pady=10)

        # Black drawing canvas
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="crosshair",
            highlightthickness=0  # removes default white border tkinter adds
        )
        self.canvas.pack(**pad)

        # Large digit label shown after prediction
        big_font = font.Font(family="Helvetica", size=72, weight="bold")
        self.result_label = tk.Label(
            self.root,
            text="?",
            font=big_font,
            fg="white",
            bg="#1a1a1a"
        )
        self.result_label.pack()

        # Confidence percentage shown underneath
        small_font = font.Font(family="Helvetica", size=14)
        self.conf_label = tk.Label(
            self.root,
            text="draw a digit",
            font=small_font,
            fg="#888888",
            bg="#1a1a1a"
        )
        self.conf_label.pack(pady=(0, 10))

        # Button row
        btn_frame = tk.Frame(self.root, bg="#1a1a1a")
        btn_frame.pack(**pad)

        btn_style = dict(
            font=font.Font(family="Helvetica", size=12),
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2"
        )

        tk.Button(
            btn_frame,
            text="Clear",
            bg="#333333",
            fg="white",
            command=self.clear,
            **btn_style
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_frame,
            text="Predict",
            bg="#4a90d9",
            fg="white",
            command=self.run_predict,
            **btn_style
        ).pack(side="left")

    def _bind_events(self):
        self.last_x = None
        self.last_y = None

        # B1-Motion fires every time mouse moves while left button held down
        self.canvas.bind("<B1-Motion>", self.paint)

        # Reset last position when mouse released so next stroke starts fresh
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)

    def paint(self, event):
        x, y = event.x, event.y

        if self.last_x is not None:
            # Draw on visible canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_SIZE,
                fill="white",
                capstyle=tk.ROUND,  # rounded ends so strokes join smoothly
                smooth=True
            )
            # Mirror exact same stroke on hidden PIL image used for prediction
            self.pil_draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, # white in greyscale
                width=BRUSH_SIZE
            )

        self.last_x = x
        self.last_y = y

    def reset_last(self, event):
        # Lift pen - next drag starts a new stroke
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all") # wipe visible canvas
        self.pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)  # wipe PIL image
        self.result_label.config(text="?")
        self.conf_label.config(text="draw a digit")

    def run_predict(self):
        arr         = preprocess_canvas(self.pil_image)  # convert drawing to (784,1) array
        digit, conf = predict(arr) # run through network
        self.result_label.config(text=str(digit))
        self.conf_label.config(text=f"{conf:.1f}% confident")


root = tk.Tk()
app  = DrawApp(root)
root.mainloop()  # starts GUI event loop - blocks until window is closed