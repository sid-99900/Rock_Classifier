import cv2
import numpy as np
import os

# ── PER-SAMPLE THRESHOLDS from your calibration ───────────────
SAMPLE_CONFIG = {
    #"basalt": {
    #    "folder" : r"C:\Users\Siddhant\Desktop\BTP_Data\dataset\basalt\basalt_sand",
    #    "lower"  : np.array([0,   0,   144]),
    #    "upper"  : np.array([179, 255, 255]),
    #},
    #"graphite": {
    #    "folder" : r"C:\Users\Siddhant\Desktop\BTP_Data\dataset\graphite\graphite_sand",
    #    "lower"  : np.array([0,   0,   140]),
    #    "upper"  : np.array([179, 255, 255]),
    #},
    #"magnetite": {
    #   "folder" : r"C:\Users\Siddhant\Desktop\BTP_Data\dataset\magnetite\magnetite_sand",
    #    "lower"  : np.array([0,   0,   156]),
    #    "upper"  : np.array([179, 255, 255]),
    #},
    #"calcite": {
    #    "folder" : r"C:\Users\Siddhant\Desktop\BTP_Data\dataset\calcite\calcite_sand",
    #    "lower"  : np.array([0,   91,  0]),
    #    "upper"  : np.array([179, 255, 255]),
    #},
    "quartz": {
        "folder" : r"C:\Users\Siddhant\Pictures\Camera Roll",
        "lower"  : np.array([0,   70,  0]),
        "upper"  : np.array([179, 255, 255]),
    },
}

CLEAN_LIMIT = 10.0   # below 10% → CLEAN
LIGHT_LIMIT = 50.0   # 10–50%   → LIGHT_SAND  |  above 50% → HEAVY_SAND
OUTPUT_FILE = r"C:\Users\Siddhant\Desktop\sand_results_all.txt"
SAVE_MASKS  = True   # saves green overlay masks for visual verification


def classify_image(img_path, lower, upper, save_mask=False):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    pct = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100

    if pct < CLEAN_LIMIT:
        label = "CLEAN"
    elif pct <= LIGHT_LIMIT:
        label = "LIGHT_SAND"
    else:
        label = "HEAVY_SAND"

    if save_mask:
        mask_dir = os.path.join(os.path.dirname(img_path), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]  # green = detected as sand
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.imwrite(
            os.path.join(mask_dir, "mask_" + os.path.basename(img_path)),
            blended)

    return pct, label


def process_all():
    all_results = []

    for sample, config in SAMPLE_CONFIG.items():
        folder = config["folder"]
        lower  = config["lower"]
        upper  = config["upper"]

        if not os.path.exists(folder):
            print(f"WARNING: {folder} not found — skipping")
            continue

        print(f"\n{'─'*55}")
        print(f"{sample.upper()}")
        print(f"{'─'*55}")

        counts = {"CLEAN": 0, "LIGHT_SAND": 0, "HEAVY_SAND": 0}

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            pct, label = classify_image(
                os.path.join(folder, filename),
                lower, upper, save_mask=SAVE_MASKS)

            if pct is None:
                continue

            counts[label] += 1
            line = f"{sample}/{filename}: {pct:.2f}% → {label}"
            all_results.append(line)
            print(f"  {filename:45s} {pct:6.2f}%  {label}")

        total = sum(counts.values())
        print(f"\n  CLEAN={counts['CLEAN']}  "
              f"LIGHT_SAND={counts['LIGHT_SAND']}  "
              f"HEAVY_SAND={counts['HEAVY_SAND']}  "
              f"(total={total})")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_results))
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all()