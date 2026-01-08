import numpy as np
from config import (
    HTC_HYPERDIFFUSION,
    KAPPA0,
    SPITZER_CONDUCTIVITY,
    HYPERTC_IN_FLUX_VECTOR,
    SATURATE_HEAT_FLUX,
    RHO,
    ENE,
    PRES,
    HEATF,
    DEFAULT_GAMMA,
    NUM_GHOST,
)

def compute_heatf_source(temperature, Q, W, S, dx, dt, cfl, max_ch, gamma=DEFAULT_GAMMA):
    if NUM_GHOST < 2:
        raise ValueError("Requires 2 ghost cells for hyperbolic tc")

    inv_dx = 1.0 / dx

    w1 = 8.0 / 12.0
    w2 = 1.0 / 12.0

    sigma_T_52 = KAPPA0
    if SPITZER_CONDUCTIVITY:
        sigma_T_52 *= temperature**2.5

    B_gradT = np.zeros_like(temperature)
    B_gradT[NUM_GHOST:-NUM_GHOST] = inv_dx * (
        w1 * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST-1:-NUM_GHOST-1]) -
        w2 * (temperature[NUM_GHOST+2:(-NUM_GHOST+2 if NUM_GHOST > 2 else None)] - temperature[NUM_GHOST-2:-NUM_GHOST-2])
    )
    sigma_T_72 = temperature * sigma_T_52
    f_sat = 1.0
    if SATURATE_HEAT_FLUX:
        f_sat = 1.0 / (1.0 + np.abs(sigma_T_52 * B_gradT) / (1.5 * W[RHO] * (gamma * W[PRES] / W[RHO])**1.5))

    tau = np.maximum(
        4.0 * dt,
        f_sat * sigma_T_72 * cfl**2 * (gamma - 1.0) / (W[PRES] * max_ch**2)
    )
    heatf_source = (f_sat * sigma_T_52 * B_gradT + W[HEATF]) / tau
    if HTC_HYPERDIFFUSION > 0.0:
        hyp = HTC_HYPERDIFFUSION / dt
        S[HEATF, NUM_GHOST:-NUM_GHOST] -= hyp * (
            (W[HEATF, NUM_GHOST+2:(-NUM_GHOST+2 if NUM_GHOST > 2 else None)] + W[HEATF, NUM_GHOST-2:-NUM_GHOST-2])
            -4.0 * (W[HEATF, NUM_GHOST+1:-NUM_GHOST+1] + W[HEATF, NUM_GHOST-1:-NUM_GHOST-1])
            + 6.0 * W[HEATF, NUM_GHOST:-NUM_GHOST]
        )
    S[HEATF, NUM_GHOST:-NUM_GHOST] -= heatf_source[NUM_GHOST:-NUM_GHOST]

    if not HYPERTC_IN_FLUX_VECTOR:
        # NOTE(cmo): Minimise additional diffusion by using 4th order FD to
        # update energy equation due to conduction
        ene_res = np.zeros_like(temperature)
        ene_res[NUM_GHOST:-NUM_GHOST] = inv_dx * (
            w1 * (W[HEATF, NUM_GHOST+1:-NUM_GHOST+1] - W[HEATF, NUM_GHOST-1:-NUM_GHOST-1]) -
            w2 * (W[HEATF, NUM_GHOST+2:(-NUM_GHOST+2 if NUM_GHOST > 2 else None)] - W[HEATF, NUM_GHOST-2:-NUM_GHOST-2])
        )
        S[ENE, NUM_GHOST:-NUM_GHOST] -= ene_res[NUM_GHOST:-NUM_GHOST]


