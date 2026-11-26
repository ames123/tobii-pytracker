from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

event_log_path = Path("./analysis_outputs/event_log.csv")
images_dir = Path("./examples/images")
out_dir = Path("./analysis_outputs/clusters")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(event_log_path)

for stim_id, df_stim in df.groupby("stimulus_id"):
    img_candidates = list(images_dir.rglob(str(stim_id)))
    if not img_candidates:
        continue

    img = Image.open(img_candidates[0])
    width, height = img.size

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)

    x_vals = df_stim["mean_x"].copy()
    y_vals = df_stim["mean_y"].copy()

    if x_vals.min() < 0 or y_vals.min() < 0:
        if x_vals.abs().max() <= 1.0 and y_vals.abs().max() <= 1.0:
            px_x = (x_vals + 0.5) * width
            px_y = (0.5 - y_vals) * height
        else:
            px_x = x_vals + (width / 2)
            px_y = (height / 2) - y_vals
    elif x_vals.max() <= 1.5 and y_vals.max() <= 1.5:
        px_x = x_vals * width
        px_y = y_vals * height
    else:
        px_x = x_vals
        px_y = y_vals

    df_plot = df_stim.copy()
    df_plot["px_x"] = px_x
    df_plot["px_y"] = px_y

    for activity, group in df_plot.groupby("activity"):
        ax.scatter(
            group["px_x"],
            group["px_y"],
            s=140,
            alpha=0.85,
            edgecolors="black",
            linewidth=1.5,
        )
        ax.text(
            group["px_x"].mean(),
            group["px_y"].mean(),
            str(activity),
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f"Obszary zainteresowania (AOI / Klastry) – {stim_id}", fontsize=13)
    ax.axis("off")

    out_file = out_dir / f"{Path(str(stim_id)).stem}_clusters.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_file}")