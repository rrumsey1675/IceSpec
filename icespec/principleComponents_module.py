import numpy as np
import copy
from spectral import principal_components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_pca(
    img_3d: np.ndarray,
    num_pcs: int = 5,
    fill_mode: str = "tile"  # "repeat_last" | "zeros" | "nan" | "tile"
) -> np.ndarray:
    """
    Run PCA on a 3D image and return an (H, W, C)-shaped array where the first
    num_pcs channels are PC scores and the remaining channels are infilled.

    fill_mode:
      - "repeat_last": repeat the last available PC into the remaining channels
      - "zeros": fill remaining channels with 0
      - "nan": fill remaining channels with NaN (use float dtype)
      - "tile": cycle/tile the PCs across remaining channels (1..k, 1..k, ...)
    """
    if img_3d.ndim != 3:
        raise ValueError(f"Expected a 3D image (H,W,C), got shape {img_3d.shape}")

    H, W, C = img_3d.shape
    k = min(num_pcs, C)

    # Deep copy to keep the original untouched
    inpaint_copy = copy.deepcopy(img_3d)

    # PCA -> get k PC score bands with shape (H, W, k)
    # pc = principal_components(inpaint_copy)
    # pc_k = pc.reduce(num=k)
    # pc_scores = pc_k.transform(inpaint_copy)  # (H, W, k)

    
    # Prepare output shaped like the original
    # Choose dtype that can handle NaN if needed
    # needs_float = fill_mode in ("nan",)
    # out_dtype = np.float32 if needs_float else pc_scores.dtype
    # out = np.empty((H, W, C), dtype=out_dtype)

    # # Place the PCs in the first k channels
    # out[:, :, :k] = pc_scores.astype(out_dtype, copy=False)
    pixel_matrix = inpaint_copy.reshape(-1, C) 
    print(f"Matrix shape (Pixels, Channels): {pixel_matrix.shape}")
    
    scaler = StandardScaler()
    pixel_matrix_scaled = scaler.fit_transform(pixel_matrix)
    
    pca = PCA(n_components=k)
    pca_results = pca.fit_transform(pixel_matrix_scaled)
    
    out = pca_results.reshape(H, W, k)

    # print(f'C: {C}; k: {k}')

    # Infill the remaining channels
    # if C > k:
    #     if fill_mode == "repeat_last":
    #         last = pc_scores[:, :, -1].astype(out_dtype, copy=False)
    #         out[:, :, k:] = last[:, :, None]
    #     elif fill_mode == "zeros":
    #         out[:, :, k:] = 0
    #     elif fill_mode == "nan":
    #         out[:, :, k:] = np.nan
    #     elif fill_mode == "tile":
    #         # Cycle the k PCs across remaining channels
    #         for i in range(k, C):
    #             print("i = ", i)
    #             out[:, :, i] = out[:, :, i % k]
    #     else:
    #         raise ValueError(f"Unknown fill_mode: {fill_mode}")

    return out
