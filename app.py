# app.py
import streamlit as st
import cv2, numpy as np, pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

# Helper to load images
def load_img(path, as_gray=False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR)
    if img is None:
        st.error(f"Failed to load {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not as_gray else img

# Sidebar navigation
st.sidebar.title("🌽 Maize Field Dashboard")
stage = st.sidebar.radio("Select View", [
    "Original & Summary",
    "1) ExG Heatmap",
    "2) Anomaly Mask",
    "3) Path Mask",
    "4) Unhealthy Patches",
    "5) U-Net Segmentation",
    "6) Canopy Cover (%)",
    "7) Grid Summary",
    "8) Time Series"
])

# Common paths
IMG_DIR    = Path("images")
OUT_DIR    = Path("output")
orig_path  = IMG_DIR / "b.png"

# Load original image once
orig_img = load_img(orig_path)

if stage == "Original & Summary":
    st.header("Original Field Image")
    st.image(orig_img, use_column_width=True)
    st.header("Combined Summary")
    summary = load_img(OUT_DIR / "summary.png")
    st.image(summary, use_column_width=True)

elif stage == "1) ExG Heatmap":
    st.header("Excess Green (ExG) Heatmap")
    heat = load_img(OUT_DIR / "exg_heatmap.png")
    st.image(heat, use_column_width=True)
    # Show raw ExG array distribution
    exg_arr = np.load(OUT_DIR / "exg_array.npy")
    st.subheader("ExG Value Distribution")
    fig, ax = plt.subplots()
    ax.hist(exg_arr.flatten(), bins=50, color='g', alpha=0.7)
    ax.set_xlabel("ExG value"); ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif stage == "2) Anomaly Mask":
    st.header("Anomalies (Very Low Green Areas)")
    mask = load_img(OUT_DIR / "anomaly_mask.png", as_gray=True)
    st.image(mask, use_column_width=True, clamp=True)

elif stage == "3) Path Mask":
    st.header("Paths & Bare Areas")
    mask = load_img(OUT_DIR / "path_mask.png", as_gray=True)
    st.image(mask, use_column_width=True, clamp=True)

elif stage == "4) Unhealthy Patches":
    st.header("Unhealthy (Stressed) Patches")
    mask = load_img(OUT_DIR / "unhealthy_mask.png", as_gray=True)
    st.image(mask, use_column_width=True, clamp=True)

elif stage == "5) U-Net Segmentation":
    st.header("U-Net Vegetation Segmentation")
    mask = load_img(OUT_DIR / "unet_mask.png", as_gray=True)
    st.image(mask, use_column_width=True, clamp=True)

elif stage == "6) Canopy Cover (%)":
    st.header("Canopy Coverage Percentage")
    # Choose which mask to use for canopy cover
    opt = st.selectbox("Mask to compute from", ["U-Net mask", "inverted Path mask"])
    if opt == "U-Net mask":
        mask = load_img(OUT_DIR / "unet_mask.png", as_gray=True)
        vegetation = mask > 0
    else:
        bare = load_img(OUT_DIR / "path_mask.png", as_gray=True)
        vegetation = bare == 0
    cover_pct = vegetation.sum() / vegetation.size * 100
    st.metric("Canopy Cover", f"{cover_pct:.2f}%")

elif stage == "7) Grid Summary":
    st.header("Grid-based ExG Summary")
    # Expect a CSV output, e.g. grid_summary.csv with columns row,col,mean_exg
    csv_path = OUT_DIR / "grid_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df)
        # Pivot to matrix for heatmap
        pivot = df.pivot(index="row", columns="col", values="mean_exg")
        fig, ax = plt.subplots()
        c = ax.imshow(pivot.values, cmap="RdYlGn")
        fig.colorbar(c, ax=ax, label="Mean ExG")
        ax.set_xlabel("Grid Col"); ax.set_ylabel("Grid Row")
        st.pyplot(fig)
    else:
        st.warning("No grid_summary.csv found in output/")

elif stage == "8) Time Series":
    st.header("Time-Series of Average ExG")
    # Expect a CSV or image plot: timeseries.png or timeseries.csv with columns time,mean_exg
    ts_csv = OUT_DIR / "timeseries.csv"
    ts_img = OUT_DIR / "timeseries.png"
    if ts_csv.exists():
        ts = pd.read_csv(ts_csv)
        fig, ax = plt.subplots()
        ax.plot(ts["time"], ts["mean_exg"], marker='o')
        ax.set_xlabel("Time"); ax.set_ylabel("Mean ExG")
        st.pyplot(fig)
        st.dataframe(ts)
    elif ts_img.exists():
        st.image(load_img(ts_img))
    else:
        st.warning("No timeseries.csv or timeseries.png found in output/")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("⚙️ Built with Streamlit")
