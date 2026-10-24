# cyst_gui.py
# Unified GUI for cyst detection:
#   1) OTSU + PNG
#   2) CLAHE (white) + DICOM
#   3) Dark (black) + DICOM

import os
import sys
import math
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ---------- helper: detect DICOM header ----------
def is_dicom_file(path):
    """True if file looks like a DICOM, even without .dcm extension."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


# ---------- OTSU + PNG ----------
def detect_cysts_on_png(
    filepath,
    area_min=3000,
    area_max=200000,
    circ_min=0.2,
    circ_max=1.2,
    blur_ksize_x=3,
    blur_ksize_y=5
):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {filepath}")

    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    kx = max(1, int(blur_ksize_x));  kx = kx + 1 if kx % 2 == 0 else kx
    ky = max(1, int(blur_ksize_y));  ky = ky + 1 if ky % 2 == 0 else ky
    blurred = cv2.GaussianBlur(img_norm, (kx, ky), 0)

    _th, binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img_norm)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
            if circ_min < circularity < circ_max:
                valid_contours.append(cnt)
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    contour_img = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    if valid_contours:
        cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 1)

    overlay = cv2.addWeighted(img_norm, 0.6, mask, 0.4, 0)
    return mask, overlay, contour_img


# ---------- CLAHE (white) + DICOM ----------
def detect_white_cyst_precise(
    dicom_path,
    min_area=2000, max_area=300000,
    lower_bright=150, upper_bright=255,
    save_csv=None, show=False
):
    import pydicom
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    try:
        px, py = map(float, ds.PixelSpacing)
    except Exception:
        px = py = 1.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
    img_eq = clahe.apply(img_norm)

    L = int(np.clip(lower_bright, 0, 255))
    U = int(np.clip(upper_bright, 0, 255))
    binary_mask = cv2.inRange(img_eq, L, U)

    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seg_mask = np.zeros_like(clean_mask, dtype=np.uint8)
    overlay  = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    objects  = []
    color = (0, 255, 0)

    idx = 1
    for cnt in contours:
        area_px = float(cv2.contourArea(cnt))
        if not (min_area < area_px < max_area):
            continue

        cv2.drawContours(seg_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(overlay, [cnt], -1, color, 2)

        area_mm2 = area_px * px * py
        eq_diam_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
        M = cv2.moments(cnt)
        cx = (M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = (M['m01'] / M['m00']) if M['m00'] != 0 else 0

        cv2.circle(overlay, (int(cx), int(cy)), 3, color, -1)
        cv2.putText(overlay, f"#{idx}", (int(cx)+6, int(cy)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        objects.append({
            "id": idx,
            "area_px": round(area_px, 1),
            "area_mm2": round(area_mm2, 2),
            "equiv_diameter_mm": round(eq_diam_mm, 2),
            "centroid_x_px": round(cx, 2),
            "centroid_y_px": round(cy, 2),
        })
        idx += 1

    segmented = cv2.bitwise_and(img_norm, img_norm, mask=seg_mask)
    return seg_mask, segmented, overlay, objects


# ---------- Dark (black) + DICOM ----------
def detect_black_cyst_precise(
    dicom_path,
    min_area=500, max_area=300000,
    lower_dark=45, upper_dark=75,
    clahe_clip=2.0, clahe_tiles=(1, 1),
    show=False
):
    import pydicom, numpy as np, cv2, math
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE (same as notebook)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
    img_eq = clahe.apply(img_norm)

    # Threshold dark band (same defaults)
    L = int(np.clip(lower_dark, 0, 255))
    U = int(np.clip(upper_dark, 0, 255))
    binary_mask = cv2.inRange(img_eq, L, U)

    # Opening with iterations=2 (matches your notebook)
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Rebuild a FILLED mask from filtered contours (like your 2nd cell)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_mask = np.zeros_like(clean_mask, dtype=np.uint8)
    keep_contours = []
    for c in contours:
        a = cv2.contourArea(c)
        if min_area < a < max_area:
            keep_contours.append(c)
            cv2.drawContours(seg_mask, [c], -1, 255, thickness=cv2.FILLED)

    # Segmented grayscale & overlay with contours (yellow so it’s distinct)
    segmented = cv2.bitwise_and(img_norm, img_norm, mask=seg_mask)
    overlay = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    if keep_contours:
        cv2.drawContours(overlay, keep_contours, -1, (0, 255, 255), 2)

    # Optional debug plot (off in GUI)
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(131); plt.imshow(img_norm, cmap='gray'); plt.title("Normalized DICOM"); plt.axis('off')
        plt.subplot(132); plt.imshow(seg_mask, cmap='gray'); plt.title("Cyst Mask (filled)"); plt.axis('off')
        plt.subplot(133); plt.imshow(segmented, cmap='gray'); plt.title("Segmented Cyst Only"); plt.axis('off')
        plt.tight_layout(); plt.show()

    # Return like other detectors: mask, overlay/segmented(gray), contour_img(BGR), objects
    # Here we use segmented as the 'overlay' pane (same as your comparison)
    return seg_mask, segmented, overlay, []  # objects empty unless you want measurements
 


# ------------------------------ GUI ------------------------------
class CystApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cyst Detector: OTSU + PNG | CLAHE (white) + DICOM | Dark (black) + DICOM")
        self.geometry("1320x760")

        self.method = tk.StringVar(value="png")  # "png" | "dcm_white" | "dcm_black"
        self.recursive = tk.BooleanVar(value=True)
        self.input_dir = tk.StringVar(value=r"C:\Users\HP\OneDrive\Desktop")

        self.files = []
        self.results = []
        self.idx = 0

        # PNG params
        self.area_min_png = tk.IntVar(value=3000)
        self.area_max_png = tk.IntVar(value=200000)
        self.circ_min_png = tk.DoubleVar(value=0.2)
        self.circ_max_png = tk.DoubleVar(value=1.2)
        self.blur_kx_png = tk.IntVar(value=3)
        self.blur_ky_png = tk.IntVar(value=5)

        # White DICOM params
        self.min_area_dcm = tk.IntVar(value=5100)
        self.max_area_dcm = tk.IntVar(value=400000)
        self.low_bright_dcm = tk.IntVar(value=160)
        self.up_bright_dcm = tk.IntVar(value=300)

        # Black DICOM params
        self.min_area_black = tk.IntVar(value=500)
        self.max_area_black = tk.IntVar(value=300000)
        self.low_dark_black = tk.IntVar(value=45)
        self.up_dark_black  = tk.IntVar(value=75)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        # Method choice
        method_box = ttk.Frame(top)
        method_box.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(method_box, text="Method: ").pack(side=tk.LEFT)
        ttk.Radiobutton(method_box, text="OTSUs + PNG", variable=self.method, value="png",
                        command=self._toggle_panels).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(method_box, text="CLAHE (white) + DICOM", variable=self.method, value="dcm_white",
                        command=self._toggle_panels).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(method_box, text="Dark (black) + DICOM", variable=self.method, value="dcm_black",
                        command=self._toggle_panels).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(method_box, text="Search subfolders", variable=self.recursive).pack(side=tk.LEFT, padx=12)

        # Folder chooser
        ttk.Label(top, text="Input folder:").grid(row=1, column=0, sticky="w")
        entry = ttk.Entry(top, textvariable=self.input_dir, width=90)
        entry.grid(row=1, column=1, padx=5, sticky="we")
        ttk.Button(top, text="Browse…", command=self.browse_folder).grid(row=1, column=2, padx=5)

        # Parameter panels
        self.params_png = self._build_png_params(top)
        self.params_dcm = self._build_dcm_params(top)
        self.params_dcm_black = self._build_dcm_black_params(top)
        self._toggle_panels()

        # Buttons
        btns = ttk.Frame(top)
        btns.grid(row=5, column=0, columnspan=3, pady=10, sticky="w")
        ttk.Button(btns, text="Load Files", command=self.load_files).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Run Detection", command=self.run_detection).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Prev", command=self.prev_image).grid(row=0, column=2, padx=5)
        ttk.Button(btns, text="Next", command=self.next_image).grid(row=0, column=3, padx=5)
        ttk.Button(btns, text="Save Current Outputs", command=self.save_current_outputs).grid(row=0, column=4, padx=5)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, padding=8).pack(side=tk.BOTTOM, fill=tk.X)

        self.fig = plt.figure(figsize=(13.2, 4.4))
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_png_params(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameters (OTSUs + PNG)")
        frame.grid(row=2, column=0, columnspan=3, pady=4, sticky="we")
        ttk.Label(frame, text="Area min").grid(row=0, column=0); ttk.Entry(frame, textvariable=self.area_min_png, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="Area max").grid(row=0, column=2); ttk.Entry(frame, textvariable=self.area_max_png, width=8).grid(row=0, column=3, padx=5)
        ttk.Label(frame, text="Circularity min").grid(row=0, column=4); ttk.Entry(frame, textvariable=self.circ_min_png, width=8).grid(row=0, column=5, padx=5)
        ttk.Label(frame, text="Circularity max").grid(row=0, column=6); ttk.Entry(frame, textvariable=self.circ_max_png, width=8).grid(row=0, column=7, padx=5)
        ttk.Label(frame, text="Blur kx").grid(row=0, column=8); ttk.Entry(frame, textvariable=self.blur_kx_png, width=6).grid(row=0, column=9, padx=5)
        ttk.Label(frame, text="Blur ky").grid(row=0, column=10); ttk.Entry(frame, textvariable=self.blur_ky_png, width=6).grid(row=0, column=11, padx=5)
        return frame

    def _build_dcm_params(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameters (CLAHE + DICOM)")
        frame.grid(row=3, column=0, columnspan=3, pady=4, sticky="we")
        ttk.Label(frame, text="Min area").grid(row=0, column=0); ttk.Entry(frame, textvariable=self.min_area_dcm, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="Max area").grid(row=0, column=2); ttk.Entry(frame, textvariable=self.max_area_dcm, width=8).grid(row=0, column=3, padx=5)
        ttk.Label(frame, text="Lower bright").grid(row=0, column=4); ttk.Entry(frame, textvariable=self.low_bright_dcm, width=8).grid(row=0, column=5, padx=5)
        ttk.Label(frame, text="Upper bright").grid(row=0, column=6); ttk.Entry(frame, textvariable=self.up_bright_dcm, width=8).grid(row=0, column=7, padx=5)
        return frame

    def _build_dcm_black_params(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameters (Dark / Black + DICOM)")
        frame.grid(row=4, column=0, columnspan=3, pady=4, sticky="we")
        ttk.Label(frame, text="Min area").grid(row=0, column=0); ttk.Entry(frame, textvariable=self.min_area_black, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="Max area").grid(row=0, column=2); ttk.Entry(frame, textvariable=self.max_area_black, width=8).grid(row=0, column=3, padx=5)
        ttk.Label(frame, text="Lower dark").grid(row=0, column=4); ttk.Entry(frame, textvariable=self.low_dark_black, width=8).grid(row=0, column=5, padx=5)
        ttk.Label(frame, text="Upper dark").grid(row=0, column=6); ttk.Entry(frame, textvariable=self.up_dark_black, width=8).grid(row=0, column=7, padx=5)
        return frame

    def _toggle_panels(self):
        if self.method.get() == "png":
            self.params_png.grid(); self.params_dcm.grid_remove(); self.params_dcm_black.grid_remove()
        elif self.method.get() == "dcm_white":
            self.params_dcm.grid(); self.params_png.grid_remove(); self.params_dcm_black.grid_remove()
        else:
            self.params_dcm_black.grid(); self.params_png.grid_remove(); self.params_dcm.grid_remove()

    def browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.input_dir.get() or os.getcwd(), title="Select input folder")
        if d: self.input_dir.set(d)

    def load_files(self):
        directory = self.input_dir.get().strip('"').strip()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select a valid folder.")
            return

        if self.method.get() == "png":
            exts = {".png"}
            def matcher(p): return os.path.splitext(p.lower())[1] in exts
        else:
            dicom_exts = {".dcm", ".dicom", ".ima"}
            def matcher(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in dicom_exts: return True
                return is_dicom_file(p)

        files = []
        if self.recursive.get():
            for root, _, names in os.walk(directory):
                for n in names:
                    path = os.path.join(root, n)
                    if os.path.isfile(path) and matcher(path):
                        files.append(path)
        else:
            for n in os.listdir(directory):
                path = os.path.join(directory, n)
                if os.path.isfile(path) and matcher(path):
                    files.append(path)

        files.sort()
        if not files:
            kind = "PNG" if self.method.get() == "png" else "DICOM"
            messagebox.showwarning("No files", f"No {kind} files found.")
            return

        self.files = files; self.results = []; self.idx = 0
        self.status.set(f"Loaded {len(self.files)} files. Ready to run.")
        self.clear_axes(); self.canvas.draw_idle()

    def run_detection(self):
        if not self.files:
            messagebox.showwarning("No files", "Load files first."); return
        self.results = []; self.status.set("Running detection…"); self.update_idletasks()

        if self.method.get() == "png":
            area_min, area_max = int(self.area_min_png.get()), int(self.area_max_png.get())
            circ_min, circ_max = float(self.circ_min_png.get()), float(self.circ_max_png.get())
            blur_kx, blur_ky = int(self.blur_kx_png.get()), int(self.blur_ky_png.get())
            for i, path in enumerate(self.files, start=1):
                try:
                    mask, overlay, contour_img = detect_cysts_on_png(path, area_min, area_max, circ_min, circ_max, blur_kx, blur_ky)
                    self.results.append({"filename": path, "mask": mask, "overlay": overlay, "contour_img": contour_img})
                except Exception as e:
                    self.results.append({"filename": path, "error": str(e)})
                self.status.set(f"Processed {i}/{len(self.files)}"); self.update_idletasks()

        elif self.method.get() == "dcm_white":
            min_area, max_area = int(self.min_area_dcm.get()), int(self.max_area_dcm.get())
            low_b, up_b = int(self.low_bright_dcm.get()), int(self.up_bright_dcm.get())
            for i, path in enumerate(self.files, start=1):
                try:
                    seg_mask, segmented, overlay_bgr, objects = detect_white_cyst_precise(path, min_area, max_area, low_b, up_b, show=False)
                    self.results.append({"filename": path, "mask": seg_mask, "overlay": segmented, "contour_img": overlay_bgr, "objects": objects})
                except Exception as e:
                    self.results.append({"filename": path, "error": str(e)})
                self.status.set(f"Processed {i}/{len(self.files)}"); self.update_idletasks()

        else:  # dcm_black
            min_area, max_area = int(self.min_area_black.get()), int(self.max_area_black.get())
            low_d, up_d = int(self.low_dark_black.get()), int(self.up_dark_black.get())
            for i, path in enumerate(self.files, start=1):
                try:
                    seg_mask, segmented, overlay_bgr, objects = detect_black_cyst_precise(path, min_area, max_area, low_d, up_d, show=False)
                    self.results.append({"filename": path, "mask": seg_mask, "overlay": segmented, "contour_img": overlay_bgr, "objects": objects})
                except Exception as e:
                    self.results.append({"filename": path, "error": str(e)})
                self.status.set(f"Processed {i}/{len(self.files)}"); self.update_idletasks()

        self.idx = 0; self.show_current(); self.status.set(f"Done. Processed {len(self.files)} files.")

    def clear_axes(self):
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()
        self.ax1.set_title("Mask"); self.ax1.axis("off")
        self.ax2.set_title("Overlay/Segmented"); self.ax2.axis("off")
        self.ax3.set_title("Contours/IDs"); self.ax3.axis("off")

    def show_current(self):
        if not self.results:
            self.clear_axes(); self.canvas.draw_idle(); return
        r = self.results[self.idx]; self.clear_axes(); title = os.path.basename(r.get("filename", f"Slice {self.idx+1}"))
        if "error" in r:
            self.ax2.text(0.5, 0.5, f"Error:\n{r['error']}", ha='center', va='center'); self.ax2.set_title(title)
        else:
            self.ax1.imshow(r["mask"], cmap="gray"); self.ax1.axis("off")
            self.ax2.imshow(r["overlay"], cmap="gray"); self.ax2.axis("off")
            self.ax3.imshow(cv2.cvtColor(r["contour_img"], cv2.COLOR_BGR2RGB)); self.ax3.axis("off")
        self.fig.suptitle(title); self.fig.tight_layout(); self.canvas.draw_idle()
        self.status.set(f"Viewing {self.idx+1}/{len(self.results)}")

    def prev_image(self):
        if not self.results: return
        self.idx = (self.idx - 1) % len(self.results); self.show_current()

    def next_image(self):
        if not self.results: return
        self.idx = (self.idx + 1) % len(self.results); self.show_current()

    def save_current_outputs(self):
        if not self.results: messagebox.showinfo("Nothing to save", "Run detection first."); return
        r = self.results[self.idx];
        if "error" in r: messagebox.showwarning("Error slice", "This slice failed; nothing to save."); return
        out_dir = filedialog.askdirectory(title="Select output folder");
        if not out_dir: return
        base = os.path.splitext(os.path.basename(r["filename"]))[0]
        try:
            cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), r["mask"])
            cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), r["overlay"])
            cv2.imwrite(os.path.join(out_dir, f"{base}_contours.png"), r["contour_img"])
            messagebox.showinfo("Saved", f"Saved {base}_*.png")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


def main():
    try: plt.switch_backend("TkAgg")
    except Exception: pass
    app = CystApp(); app.mainloop()


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print("Fatal error:", exc, file=sys.stderr); raise
