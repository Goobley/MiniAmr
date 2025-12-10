from typing import Tuple
import numpy as np
Array = np.ndarray

def reconstruct_fog(W: Array) -> Tuple[Array, Array]:
    return (W.copy(), W.copy())

def slope_limiter(um, up):
    # MC
    return (np.copysign(1.0, um) + np.copysign(1.0, up)) * np.minimum(
        np.abs(um),
        np.minimum(
            0.25 * np.abs(um + up),
            np.abs(up)
        )
    )

def reconstruct_plm_nonuniform(W: Array, x_cen: Array, cell_vol: Array) -> Tuple[Array, Array]:
    # Left edge of cell
    WL = W.copy()
    # Right edge of cell
    WR = W.copy()

    dxL = x_cen[1:-1] - x_cen[:-2]
    dxR = x_cen[2:] - x_cen[1:-1]
    dwL = W[:, 1:-1] - W[:, :-2]
    dwR = W[:, 2:] - W[:, 1:-1]

    # slopes
    sL = dwL / dxL
    sR = dwR / dxR
    delta = slope_limiter(sL, sR)

    dx_centre = 0.5 * cell_vol[1:-1]
    WL[:, 1:-1] = WL[:, 1:-1] - dx_centre * delta
    WR[:, 1:-1] = WR[:, 1:-1] + dx_centre * delta
    return WL, WR

def reconstruct_plm(W: Array) -> Tuple[Array, Array]:
    # Left edge of cell
    WL = W.copy()
    # Right edge of cell
    WR = W.copy()

    dwL = W[:, 1:-1] - W[:, :-2]
    dwR = W[:, 2:] - W[:, 1:-1]

    # slopes
    delta = slope_limiter(dwL, dwR)

    WL[:, 1:-1] = WL[:, 1:-1] - 0.5 * delta
    WR[:, 1:-1] = WR[:, 1:-1] + 0.5 * delta
    return WL, WR