import numpy as np
Array = np.ndarray
from config import DEFAULT_GAMMA, RHO, MOM, ENE, VEL, PRES, HEATF, HYPERTC_IN_FLUX_VECTOR, USE_CONDUCTION, CONDUCTION_ONLY, k_B

def cons_to_prim(Q: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    W = np.empty_like(Q)

    rho = Q[RHO, :]
    mom = Q[MOM, :]
    E = Q[ENE, :]

    v = mom / rho
    kinetic = 0.5 * rho * v**2
    e = E - kinetic
    p = (gamma - 1.0) * e

    W[RHO] = rho
    W[VEL] = v
    W[PRES] = p
    if USE_CONDUCTION:
        W[HEATF] = Q[HEATF]

    return W

def prim_to_cons(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    Q = np.empty_like(W)

    rho = W[RHO, :]
    v = W[VEL, :]
    p = W[PRES, :]

    mom = rho * v
    energy = p / (gamma - 1.0) + 0.5 * mom**2 / rho

    Q[RHO] = rho
    Q[MOM] = mom
    Q[ENE] = energy
    if USE_CONDUCTION:
        Q[HEATF] = W[HEATF]

    return Q

def prim_to_flux(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    flux = np.empty_like(W)
    rho = W[RHO, :]
    v = W[VEL, :]
    p = W[PRES, :]

    mass_flux = rho * v
    mom_flux = mass_flux * v + p

    e_kin = 0.5 * rho * v**2
    e_tot = p / (gamma - 1.0) + e_kin
    ene_flux = (e_tot + p) * v

    if USE_CONDUCTION:
        flux[HEATF] = 0.0
        if HYPERTC_IN_FLUX_VECTOR:
            ene_flux += W[HEATF]
        if CONDUCTION_ONLY:
            mass_flux = 0.0
            mom_flux = 0.0
            ene_flux = 0.0
            if HYPERTC_IN_FLUX_VECTOR:
                ene_flux = W[HEATF]

    flux[RHO] = mass_flux
    flux[MOM] = mom_flux
    flux[ENE] = ene_flux

    return flux

def sound_speed(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    rho = W[RHO]
    p = W[PRES]
    return np.sqrt(gamma * p / rho)

def rusanov_flux(WL: Array, WR: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    vL = WL[VEL]
    vR = WR[VEL]

    qL = prim_to_cons(WL, gamma)
    qR = prim_to_cons(WR, gamma)

    fL = prim_to_flux(WL, gamma)
    fR = prim_to_flux(WR, gamma)

    csL = sound_speed(WL, gamma)
    csR = sound_speed(WR, gamma)

    max_c = 0.5 * (csL + np.abs(vL) + csR + np.abs(vR))

    flux = 0.5 * (fL + fR - max_c * (qR - qL))
    return flux

def temperature_si(pressure, n_baryon, y=1.0):
    return pressure / (n_baryon * (1.0 + y) * k_B)