from typing import TYPE_CHECKING

import lineax
from config import RHO, ENE, k_B, P_MASS, MEAN_MOLECULAR_MASS, DEFAULT_GAMMA, KAPPA0, NUM_GHOST
import jax
import jax.numpy as jnp
import optimistix as optx

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from main import Forest

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)

def residual(temperature, xcc, x_int, kappa0=KAPPA0, alpha=1.0, beta=2.5):
    resid = jnp.zeros_like(temperature)
    kappa = alpha * kappa0 * temperature**beta
    kappa_if = jnp.zeros(resid.shape[0] + 1)
    kappa_if = kappa_if.at[1:-1].set(harmonic_mean(kappa[:-1], kappa[1:]))

    dx = x_int[1:] - x_int[:-1]
    resid = resid.at[NUM_GHOST:-NUM_GHOST].set(
        1.0 / dx[NUM_GHOST:-NUM_GHOST] * (
            kappa_if[NUM_GHOST+1:-NUM_GHOST] *
                (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST:-NUM_GHOST]) /
                (xcc[NUM_GHOST+1:-NUM_GHOST+1] - xcc[NUM_GHOST:-NUM_GHOST])
            - kappa_if[NUM_GHOST:-NUM_GHOST-1] *
                (temperature[NUM_GHOST:-NUM_GHOST] - temperature[NUM_GHOST-1:-NUM_GHOST-1]) /
                (xcc[NUM_GHOST:-NUM_GHOST] - xcc[NUM_GHOST-1:-NUM_GHOST-1])
        )
    )
    return resid

def implicit_residual(temperature, prev_temperature, xcc, x_int, dt, kappa0=KAPPA0, alpha=1.0, beta=2.5, theta=0.55):
    r_new = residual(temperature, xcc, x_int, kappa0=kappa0, alpha=alpha, beta=beta)
    r_old = residual(prev_temperature, xcc, x_int, kappa0=kappa0, alpha=alpha, beta=beta)
    return temperature - prev_temperature - dt * (theta * r_new + (1.0 - theta) * r_old)

@jax.jit
def solve_step(prev_temperature, xcc, x_int, dt, alpha):
    def to_opt(y, args):
        return implicit_residual(y, prev_temperature, xcc, x_int, dt, alpha=alpha)

    sol = optx.root_find(
        to_opt,
        solver=optx.Newton(
            rtol=1e-5,
            atol=1e-5,
            linear_solver=lineax.GMRES(rtol=1e-5, atol=1e-5)),
        y0=prev_temperature,
    )
    print(sol.stats)
    return sol.value

def implicit_thermal_conduction(
    forest: "Forest",
    dt,
    gamma=DEFAULT_GAMMA
):
    xcc, x_int, Q = forest.flatten(return_interfaces=True)
    # TODO(cmo): This implicitly assumes fully ionised only H
    cv = 1.0 / (gamma - 1.0) * k_B / (P_MASS * MEAN_MOLECULAR_MASS)
    # HACK(cmo): Because this doesn't pick up changes made to config (it has already imported).
    cv = 1.0 / (gamma - 1.0)
    # cv = 1.0
    alpha = 1.0 / (Q[RHO] * cv)
    temperature = Q[ENE] * alpha

    # NOTE(cmo): Hack here at alpha too -- should be alpha=alpha
    new_temperature = solve_step(temperature, xcc, x_int, dt, alpha=1.0)
    delta_E = (new_temperature - temperature) / alpha
    Q[ENE] += delta_E
    Q_slice = Q[:, NUM_GHOST:-NUM_GHOST]
    leaves = forest.get_leaves()
    block_size = forest.M
    for i, leaf in enumerate(leaves):
        leaf.Q[ENE, NUM_GHOST:-NUM_GHOST] = Q_slice[ENE, i*block_size:(i+1)*block_size]
