import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def fill_mekong_tensor_numba(feature_tensor, count_tensor, t_indices,
                             x_starts, x_ends, y_starts, y_ends, feat_vals):
    n_rows = len(t_indices)

    for i in prange(n_rows):
        t = t_indices[i]
        x0, x1 = x_starts[i], x_ends[i]
        y0, y1 = y_starts[i], y_ends[i]

        for y in range(y0, y1):
            for x in range(x0, x1):
                for c in range(feat_vals.shape[1]):
                    feature_tensor[t, y, x, c] += feat_vals[i, c]
                    count_tensor[t, y, x, c] += 1

    return feature_tensor, count_tensor


@jit(nopython=True)
def compute_sentinel_slice(dates_array, target_date, x_starts, x_ends,
                          y_starts, y_ends, feat_vals, H, W, C):
    best_td = np.full((H, W, C), np.inf, dtype=np.float32)
    best_sum = np.zeros((H, W, C), dtype=np.float32)
    best_count = np.zeros((H, W, C), dtype=np.int32)

    n_rows = len(dates_array)

    for i in range(n_rows):
        td = abs(dates_array[i] - target_date)
        x0, x1 = x_starts[i], x_ends[i]
        y0, y1 = y_starts[i], y_ends[i]

        for y in range(y0, y1):
            for x in range(x0, x1):
                for c in range(C):
                    v = feat_vals[i, c]
                    if not np.isnan(v):
                        if td < best_td[y, x, c]:
                            best_td[y, x, c] = td
                            best_sum[y, x, c] = v
                            best_count[y, x, c] = 1
                        elif abs(td - best_td[y, x, c]) < 0.001:
                            best_sum[y, x, c] += v
                            best_count[y, x, c] += 1

    result = np.zeros((H, W, C), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            for c in range(C):
                if best_count[y, x, c] > 0:
                    result[y, x, c] = best_sum[y, x, c] / best_count[y, x, c]

    return result