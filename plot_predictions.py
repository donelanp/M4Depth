import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def normalize_depth(d):
    """Normalize depth for visualization."""
    d = np.squeeze(d)
    d = np.nan_to_num(d)
    if d.max() > d.min():
        d = (d - d.min()) / (d.max() - d.min())
    return d


def plot_sample(npz_path, output_path):
    data = np.load(npz_path)

    rgb = data["rgb"]                # [H,W,3]
    d_est = data["depth_est"]        # [H,W,1]
    d_gt = data["depth_gt"]          # [H,W,1]

    # Normalize for visualization
    d_est_norm = normalize_depth(d_est)
    d_gt_norm = normalize_depth(d_gt)
    error_map = np.abs(d_est - d_gt)
    error_norm = normalize_depth(error_map)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(rgb.astype(np.float32))
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    axes[1].imshow(d_est_norm, cmap="inferno")
    axes[1].set_title("Predicted Depth")
    axes[1].axis("off")

    axes[2].imshow(d_gt_norm, cmap="inferno")
    axes[2].set_title("Ground Truth Depth")
    axes[2].axis("off")

    axes[3].imshow(error_norm, cmap="magma")
    axes[3].set_title("Abs Error Map")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="saved_predictions")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    files.sort()

    print(f"Found {len(files)} prediction samples.")

    for i, f in enumerate(files):
        in_path = os.path.join(args.input_dir, f)
        out_path = os.path.join(args.output_dir, f.replace(".npz", ".png"))
        print(f"[{i+1}/{len(files)}] Plotting {out_path}")
        plot_sample(in_path, out_path)

    print("Done! Plots saved in:", args.output_dir)


if __name__ == "__main__":
    main()
