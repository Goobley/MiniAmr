import numpy as np
Array = np.ndarray

DEFAULT_GAMMA = 1.4

def cons_to_prim(Q: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    rho = Q[0, :]
    mom = Q[1, :]
    E = Q[2, :]

    v = mom / rho
    kinetic = 0.5 * rho * v**2
    e = E - kinetic
    p = (gamma - 1.0) * e
    return np.stack([rho, v, p])

def prim_to_cons(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    rho = W[0, :]
    v = W[1, :]
    p = W[2, :]

    mom = rho * v
    energy = p / (gamma - 1.0) + 0.5 * mom**2 / rho
    return np.stack([rho, mom, energy])

def prim_to_flux(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    rho = W[0, :]
    v = W[1, :]
    p = W[2, :]

    mass_flux = rho * v
    mom_flux = mass_flux * v + p

    e_kin = 0.5 * rho * v**2
    e_tot = p / (gamma - 1.0) + e_kin
    ene_flux = (e_tot + p) * v
    return np.stack([mass_flux, mom_flux, ene_flux])

def sound_speed(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    rho = W[0]
    p = W[2]
    return np.sqrt(gamma * p / rho)

def rusanov_flux(WL: Array, WR: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    vL = WL[1]
    vR = WR[1]

    qL = prim_to_cons(WL, gamma)
    qR = prim_to_cons(WR, gamma)

    fL = prim_to_flux(WL, gamma)
    fR = prim_to_flux(WR, gamma)

    csL = sound_speed(WL, gamma)
    csR = sound_speed(WR, gamma)

    max_c = 0.5 * (csL + np.abs(vL) + csR + np.abs(vR))

    flux = 0.5 * (fL + fR - max_c * (qR - qL))
    return flux
