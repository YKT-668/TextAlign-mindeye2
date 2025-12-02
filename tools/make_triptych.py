import os
import json
import glob
import math
import torch
from PIL import Image

# --- Configuration ---
base = "/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference"
gt_all_path = "/home/vipuser/MindEyeV2_Project/src/evals/all_images.pt"
out_png = os.environ.get("OUT_PNG", f"{base}/triptych_default.png")
ROWS = int(os.environ.get("ROWS", 2))
COLS = int(os.environ.get("COLS", 5))

def main():
    # --- Gracefully handle missing ids.json ---
    ids_path = f"{base}/ids.json"
    ids = []
    if os.path.exists(ids_path):
        try:
            with open(ids_path, 'r') as f:
                ids = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load or parse {ids_path}. Using sequential indices for GT images.")
    else:
        print(f"Warning: {ids_path} not found. Using sequential indices for GT images.")

    # --- Find all image files ---
    enhanced_files = sorted(glob.glob(f"{base}/enhanced/enhanced_*.png"))
    recon_files = sorted(glob.glob(f"{base}/images/*.png"))

    if not enhanced_files or not recon_files:
        print("Error: No enhanced or reconstructed images found in their respective directories. Aborting.")
        return

    # --- Load Ground Truth (GT) images ---
    try:
        gt_images_tensor = torch.load(gt_all_path, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_all_path}. Aborting.")
        return

    def tensor_to_pil(t):
        # Denormalize if necessary and convert to PIL Image
        if t.max() <= 1.0:
            t = t * 255.0
        t = t.clamp(0, 255)
        return Image.fromarray(t.permute(1, 2, 0).to(torch.uint8).numpy())

    # Determine the number of samples to process
    num_samples = min(len(enhanced_files), len(recon_files), ROWS * COLS)
    if num_samples == 0:
        print("No images to process for the triptych.")
        return

    print(f"Found {len(gt_images_tensor)} GT images, {len(recon_files)} reconstructions, {len(enhanced_files)} enhanced images.")
    print(f"Creating a triptych of {num_samples} samples in a grid up to {ROWS}x{COLS}.")

    cards = []
    for i in range(num_samples):
        # Use loaded IDs if available and valid, otherwise fall back to sequential index
        gt_index = ids[i] if i < len(ids) else i
        if gt_index >= len(gt_images_tensor):
            print(f"Warning: Index {gt_index} is out of bounds for GT images (total: {len(gt_images_tensor)}). Skipping sample {i}.")
            continue

        # Load the three images for the current sample
        gt_im_tensor = gt_images_tensor[gt_index]
        gt_im = tensor_to_pil(gt_im_tensor)
        
        rec_im = Image.open(recon_files[i]).convert("RGB")
        enh_im = Image.open(enhanced_files[i]).convert("RGB")
        
        # Resize all to a common size for neat stacking
        target_size = (256, 256)
        gt_im = gt_im.resize(target_size)
        rec_im = rec_im.resize(target_size)
        enh_im = enh_im.resize(target_size)
        
        # Create a vertical strip (card) for one sample: GT, Recon, Enhanced
        card = Image.new("RGB", (target_size[0], target_size[1] * 3), (255, 255, 255))
        card.paste(gt_im, (0, 0))
        card.paste(rec_im, (0, target_size[1]))
        card.paste(enh_im, (0, target_size[1] * 2))
        cards.append(card)

    if not cards:
        print("No valid image cards were created. Exiting.")
        return

    # --- Assemble the final grid image ---
    actual_num_cards = len(cards)
    final_cols = min(COLS, actual_num_cards)
    final_rows = math.ceil(actual_num_cards / final_cols)
    
    canvas_width = target_size[0] * final_cols
    canvas_height = (target_size[1] * 3) * final_rows

    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    for idx, c in enumerate(cards):
        row_idx = idx // final_cols
        col_idx = idx % final_cols
        canvas.paste(c, (col_idx * target_size[0], row_idx * (target_size[1] * 3)))

    canvas.save(out_png)
    print(f"SUCCESS: Triptych saved to -> {out_png}")

if __name__ == "__main__":
    main()
