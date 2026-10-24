import os
import uuid
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from unet_model import UNet

# -------------------
# CONFIGURE PATHS
# -------------------
# Change these two paths to your actual model files:
YOLO_WEIGHTS = "best.pt"
UNET_WEIGHTS = "unet_seg.pth"

# CLASSES (matching your training script)
CLASSES = ["Background", "Abscess", "Cyst", "Tumor"]  # 0..3
NUM_CLASSES = len(CLASSES)

# Image size used for UNet input
IMG_SIZE = 256

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Flask
app = Flask(__name__)
app.secret_key = "seg@secret"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_name(name: str) -> str:
    stem = os.path.splitext(secure_filename(name))[0]
    return f"{uuid.uuid4().hex}_{stem}"


def save_image_bgr(img_bgr: np.ndarray, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img_bgr)
    return path


def overlay_on_image(img_bgr: np.ndarray, mask_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend only where mask has non-zero intensity."""
    overlay = img_bgr.copy()
    mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    idx = mask_gray > 0
    overlay[idx] = (alpha * overlay[idx] + (1 - alpha) * mask_bgr[idx]).astype(np.uint8)
    return overlay


# -------------------
# LOAD MODELS
# -------------------
# YOLOv8 Segmentation
try:
    yolo_model = YOLO(YOLO_WEIGHTS)
except Exception as e:
    yolo_model = None
    print(f"[WARN] Could not load YOLO weights at {YOLO_WEIGHTS}: {e}")

# UNet (exact architecture from your training code)
try:
    unet_model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    unet_model.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))
    unet_model.eval()
except Exception as e:
    unet_model = None
    print(f"[WARN] Could not load UNet weights at {UNET_WEIGHTS}: {e}")


# -------------------
# ROUTES
# -------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/segment/<method>", methods=["GET", "POST"])
def segment(method):
    method = method.lower()
    if method not in {"yolo", "unet"}:
        flash("Unknown method requested.")
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("segment.html", method=method, orig=None)

    # POST: handle upload
    file = request.files.get("image")
    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please choose a valid image file.")
        return redirect(request.url)

    # Save upload
    base = unique_name(file.filename)
    upload_name = base + ".png"
    upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
    file.save(upload_path)

    # Read image
    img_bgr = cv2.imread(upload_path)
    if img_bgr is None:
        flash("Failed to read the uploaded image.")
        return redirect(request.url)
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Prepare outputs
    classes_present = set()
    mask_color = np.zeros((H, W, 3), dtype=np.uint8)

    if method == "yolo":
        if yolo_model is None:
            flash("YOLO model not loaded. Check YOLO_WEIGHTS path.")
            return redirect(request.url)

        with torch.no_grad():
            results = yolo_model(img_rgb)

        # Build a class->color palette (BGR) that stays consistent
        name_to_color = {}

        def color_for(name: str):
            if name in name_to_color:
                return name_to_color[name]
            # deterministic pastel-like color from hash
            h = abs(hash(name)) % 255
            bgr = (int((h * 97) % 256), int((h * 57) % 256), int((h * 23) % 256))
            name_to_color[name] = bgr
            return bgr

        overlay = img_bgr.copy()

        if len(results) > 0 and results[0].masks is not None:
            res = results[0]
            names = getattr(res, "names", None)
            for mask_tensor, box in zip(res.masks.data, res.boxes):
                # mask_tensor is HxW (normalized 0..1), resize to image
                m = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                cls_id = int(box.cls.cpu().numpy())
                cls_name = names[cls_id] if names is not None and cls_id in names else str(cls_id)
                classes_present.add(cls_name)
                bgr = color_for(cls_name)

                # paint mask_color where mask==1
                for c in range(3):
                    channel = mask_color[:, :, c]
                    channel[m == 1] = bgr[c]
                    mask_color[:, :, c] = channel

                # draw bbox + label on overlay
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    overlay,
                    cls_name,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
        else:
            overlay = img_bgr.copy()

        # Final overlay with colored mask
        overlay = overlay_on_image(overlay, mask_color, alpha=0.45)

    else:  # method == "unet"
        if unet_model is None:
            flash("UNet model not loaded. Check UNET_WEIGHTS path.")
            return redirect(request.url)

        # Preprocess for UNet
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        chw = np.transpose(img_resized, (2, 0, 1))[None, ...]  # 1x3xHxW
        with torch.no_grad():
            tensor = torch.from_numpy(chw).to(DEVICE)
            logits = unet_model(tensor)              # [1, C, H, W]
            pred_class = torch.argmax(logits, dim=1) # [1, H, W]
            pred_class = pred_class.squeeze(0).cpu().numpy().astype(np.uint8)

        # Upsample to original size
        class_map = cv2.resize(pred_class, (W, H), interpolation=cv2.INTER_NEAREST)

        # Palette for indices 0..3 (BGR)
        palette = {
            0: (0, 0, 0),       # Background
            1: (0, 0, 255),     # Abscess -> Red (B,G,R) => Red is last channel
            2: (0, 255, 0),     # Cyst -> Green
            3: (255, 0, 0),     # Tumor -> Blue
        }
        # Colorize mask
        for idx in np.unique(class_map):
            bgr = palette.get(int(idx), (200, 200, 200))
            mask_color[class_map == idx] = bgr
            if 0 <= idx < len(CLASSES):
                classes_present.add(CLASSES[idx])

        overlay = overlay_on_image(img_bgr, mask_color, alpha=0.45)

    # Save outputs
    orig_name = base + "_orig.jpg"
    mask_name = base + "_mask.png"
    overlay_name = base + "_overlay.jpg"

    # original should be RGB->BGR for saving
    save_image_bgr(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), RESULT_FOLDER, orig_name)
    save_image_bgr(mask_color, RESULT_FOLDER, mask_name)
    save_image_bgr(overlay, RESULT_FOLDER, overlay_name)

    # Build URLs for template
    orig_rel = f"results/{orig_name}"
    mask_rel = f"results/{mask_name}"
    overlay_rel = f"results/{overlay_name}"

    # Show legend in result
    classes_list = sorted(list(classes_present))
    return render_template(
        "segment.html",
        method=method,
        orig=orig_rel,
        mask=mask_rel,
        overlay=overlay_rel,
        classes=classes_list,
    )


if __name__ == "__main__":
    # For Windows: set threaded=True to avoid reloader file lock quirks
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
