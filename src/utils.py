import numpy as np
import torch
import matplotlib.pyplot as plt
import geopandas as gpd

def autoregressive_forecast(model, initial_seq, n_days=7, device='cuda'):
    """
    Sinh dự báo n bước tự hồi quy (autoregressive)
    initial_seq: [1, time_in, C, H, W]
    """
    model.eval()
    current_seq = initial_seq.to(device)
    preds = []

    with torch.no_grad():
        for _ in range(n_days):
            out = model(current_seq)  # [1,1,C,H,W]
            preds.append(out[0,0].cpu().numpy())
            current_seq = torch.cat([current_seq[:,1:], out], dim=1)
    return np.array(preds)  # [n_days,C,H,W]

# ================================================================
# 🗺️ Visualization
# ================================================================
def plot_map(
    rain_map, title, vn_adm1, dbscl, min_lon=103.5, max_lon=107, min_lat=8, max_lat=11.5
):
    """
    Hiển thị bản đồ dự đoán lượng mưa trên vùng ĐBSCL.
    Trả về figure để Streamlit dùng st.pyplot(fig)
    """
    extent = [min_lon, max_lon, min_lat, max_lat]
    fig, ax = plt.subplots(figsize=(10, 10))

    # Ranh giới hành chính
    vn_adm1.boundary.plot(ax=ax, color="lightgrey", linewidth=0.4)
    dbscl.boundary.plot(ax=ax, color="black", linewidth=1)

    # Overlay ma trận dự đoán
    im = ax.imshow(rain_map, extent=extent, origin="lower", cmap="turbo", alpha=0.8)

    # Tùy chỉnh hiển thị
    fig.colorbar(im, ax=ax, label="Rainfall prediction (mm)")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Kinh độ")
    ax.set_ylabel("Vĩ độ")
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    plt.tight_layout()

    return fig  # ✅ Trả về figure, không dùng plt.show()



# ================================================================
# 🔁 Forecasting Functions
# ================================================================
def autoregressive_forecast_last(model, initial_seq, n_days=7, device="cuda"):
    """
    Dự đoán đặc trưng (feature maps) cho n_days tiếp theo
    bằng phương pháp autoregressive.
    """
    model.eval()
    current_seq = initial_seq.to(device)
    preds = []

    with torch.no_grad():
        for _ in range(n_days):
            out = model(current_seq)  # [1,1,C,H,W]
            preds.append(out[0, 0].cpu().numpy())
            current_seq = torch.cat([current_seq[:, 1:], out], dim=1)

    return np.array(preds)[-1]  # lấy ngày cuối cùng


def predict_rainfall(model, data, minmax, robust, device="cuda"):
    """
    Dự đoán lượng mưa (mm/ngày) từ feature maps đã được dự đoán.
    Tự động inverse scaling bằng MinMax + Robust.
    """
    model.eval()
    with torch.no_grad():
        preds = model(data)
        N, C, H, W = preds.shape

        preds_np = preds.detach().cpu().numpy().reshape(-1, 1)
        preds_inv = robust.inverse_transform(
            minmax.inverse_transform(preds_np)
        ).reshape(N, C, H, W)

    return preds_inv[0, 0]  # [H, W]