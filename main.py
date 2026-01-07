from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from hyperbolic_thermal_conduction import compute_heatf_source
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

import numpy as np
from euler import DEFAULT_GAMMA, cons_to_prim, prim_to_cons, prim_to_flux, rusanov_flux, sound_speed, temperature_si
from reconstruct import reconstruct_fog, reconstruct_plm, slope_limiter

from config import *

class Block:
    def __init__(self, depth, max_depth, M, x0, x1, parent=None):
        self.depth = depth
        self.max_depth = max_depth
        self.M = M
        self.x0 = x0
        self.x1 = x1
        self.dx = (x1 - x0) / M

        self.Q = np.zeros((NUM_EQ, M + 2 * NUM_GHOST))

        # Child blocks (None if leaf)
        self.left: Optional[Block] = None
        self.right: Optional[Block] = None
        self.parent: Optional[Block] = parent

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def x_pos(self):
        return (self.x0 - NUM_GHOST * self.dx) + np.arange(self.Q.shape[1]) * self.dx

    def refine(self):
        if self.depth == self.max_depth:
            return

        xm = 0.5 * (self.x0 + self.x1)

        self.left = Block(self.depth+1, self.max_depth, self.M, self.x0, xm, parent=self)
        self.right = Block(self.depth+1, self.max_depth, self.M, xm, self.x1, parent=self)

        half_grid = self.Q.shape[1] // 2

        # Prolongation -- ghosts aren't handled correctly, but that's fine, they
        # get filled before each step and also before CFL calculation
        duL = self.Q[:, 1:-1] - self.Q[:, :-2]
        duR = self.Q[:, 2:] - self.Q[:, 1:-1]
        delta = np.zeros_like(self.Q)
        delta[:, 1:-1] = slope_limiter(duL, duR)

        self.left.Q[:, NUM_GHOST:-NUM_GHOST:2] = (
            self.Q[:, NUM_GHOST:half_grid] - 0.25 * delta[:, NUM_GHOST:half_grid]
        )
        self.left.Q[:, NUM_GHOST+1:-NUM_GHOST:2] = (
            self.Q[:, NUM_GHOST:half_grid] + 0.25 * delta[:, NUM_GHOST:half_grid]
        )

        self.right.Q[:, NUM_GHOST:-NUM_GHOST:2] = (
            self.Q[:, half_grid:-NUM_GHOST] - 0.25 * delta[:, half_grid:-NUM_GHOST]
        )
        self.right.Q[:, NUM_GHOST+1:-NUM_GHOST:2] = (
            self.Q[:, half_grid:-NUM_GHOST] + 0.25 * delta[:, half_grid:-NUM_GHOST]
        )

    def coarsen(self):
        if self.is_leaf:
            return

        # Simplest restriction -- ignore ghosts. Rebuild grid and local average
        self.Q[:, NUM_GHOST:-NUM_GHOST] = 0.5 * np.concat(
            [
                self.left.Q[:, NUM_GHOST:-NUM_GHOST],
                self.right.Q[:, NUM_GHOST:-NUM_GHOST],
            ],
            axis=1,
        ).reshape(NUM_EQ, -1, 2).sum(axis=2)

        self.left = None
        self.right = None

    def get_leaves(self):
        if self.is_leaf:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    def lohner_indicator(self, eps=1e-12):
        rho = self.Q[0]

        d2 = np.abs(rho[:-2] - 2*rho[1:-1] + rho[2:])
        d1 = np.abs(rho[2:] - rho[1:-1]) + np.abs(rho[1:-1] - rho[:-2])
        return np.max(d2 / (d1 + eps))

    def get_dt_ch(self, max_cfl: float = 0.6, gamma: float=DEFAULT_GAMMA):
        """
        Computes the min dt and max propagation speed for the block
        """
        w = cons_to_prim(self.Q)
        cs = sound_speed(w, gamma=gamma)
        fast_speed = np.abs(w[1]) + cs

        dt_local = max_cfl * self.dx / fast_speed
        return np.min(dt_local), np.max(fast_speed)

class Forest:
    def __init__(self, x0, x1, N=4, M=64, L=2):
        """
        :param x0: min x
        :param x1: max x
        :param N: number of trees (blocks)
        :param M: cells per block
        :param L: max refinement level (inclusive, 0 indexed)
        """
        self.N = N
        self.M = M
        self.L = L

        block_interfaces = np.linspace(x0, x1, N+1)
        self.roots: List[Block] = []
        for i in range(N):
            block = Block(
                depth=0,
                max_depth=L,
                M=M,
                x0=block_interfaces[i],
                x1=block_interfaces[i+1],
                parent=None
            )
            self.roots.append(block)
        self.fixed_bcs = np.zeros((2, NUM_EQ))
        self.left_bc_fn = lambda block, dt: None
        self.right_bc_fn = lambda block, dt: None

    def get_leaves(self) -> List[Block]:
        result = []

        for root in self.roots:
            result += root.get_leaves()

        return result

    def ensure_2to1_balance(self):
        leaves = self.get_leaves()
        for i in range(len(leaves)-1):
            if abs(leaves[i].depth - leaves[i+1].depth) > 1:
                raise ValueError("Forest not properly balanced!")

    def regrid(self, refine_threshold, deref_ratio):
        leaves = self.get_leaves()
        flags = {leaf: 0 for leaf in leaves}

        # NOTE(cmo): Compute block indicators
        for leaf in leaves:
            err = leaf.lohner_indicator()
            if err > refine_threshold:
                flags[leaf] = 1
            if err < refine_threshold * deref_ratio:
                flags[leaf] = -1

        # NOTE(cmo): Remove coarsen flag from blocks that can't be coarsened (as
        # sibling not flagged for it). Now, any block that has a coarsen flag
        # _can_ be coarsened, provided it ensures 2:1 balance. Also remove any
        # coarsen on level 0.
        for leaf in leaves:
            if flags[leaf] != -1:
                continue

            if leaf.depth == 0:
                flags[leaf] = 0
            # Don't coarsen if sibling isn't a leaf
            elif not leaf.parent.left.is_leaf or not leaf.parent.right.is_leaf:
                flags[leaf] = 0
            # Don't coarsen unless both children (which are both leaves) want to coarsen
            elif flags[leaf.parent.left] != -1 or flags[leaf.parent.right] != -1:
                flags[leaf] = 0

        # NOTE(cmo): Enforce 2:1 balance
        changed = True
        while changed:
            changed = False
            for i, leaf in enumerate(leaves):
                neighbours = [
                    leaves[i-1] if i > 0 else None,
                    leaves[i+1] if i < len(leaves) - 1 else None
                ]
                neighbours = [n for n in neighbours if n is not None]

                # NOTE(cmo): Do we need to refine ourselves to stay balanced with out neighbours?
                for neighbour in neighbours:
                    if leaf.depth + flags[leaf] < neighbour.depth + flags[neighbour] - 1:
                        if flags[leaf] != 1:
                            flags[leaf] += 1
                            changed = True

            if changed:
                # NOTE(cmo): Ensure potential coarsens are preserved on the leaves
                for leaf in leaves:
                    if flags[leaf] != -1:
                        continue

                    # Don't coarsen if sibling isn't a leaf
                    if not leaf.parent.left.is_leaf or not leaf.parent.right.is_leaf:
                        flags[leaf] = 0
                    # Don't coarsen unless both children (which are both leaves) want to coarsen
                    elif flags[leaf.parent.left] != -1 or flags[leaf.parent.right] != -1:
                        flags[leaf] = 0

        # NOTE(cmo): Apply refinement
        for leaf in leaves:
            if flags[leaf] == 1:
                leaf.refine()

        # NOTE(cmo): Apply coarsening
        blocks_visited = {}
        for leaf in leaves:
            if flags[leaf] == -1:
                if leaf in blocks_visited:
                    continue

                left = leaf.parent.left
                right = leaf.parent.right
                if flags[left] == -1 and flags[right] == -1:
                    leaf.parent.coarsen()
                    blocks_visited[left] = True
                    blocks_visited[right] = True

        self.ensure_2to1_balance()

    def set_ics(self, ics, gamma: float=DEFAULT_GAMMA):
        for i in range(self.L + 1):
            print(f"Set ICS for level {i}")
            for leaf in self.get_leaves():
                leaf.Q[:] = ics(leaf.x_pos, gamma=gamma)

            self.regrid(refine_threshold=ERR_THRESH, deref_ratio=DEREF_RATIO)

    def set_fixed_bcs(self, bcs):
        self.fixed_bcs[0] = bcs[0]
        self.fixed_bcs[1] = bcs[1]

    def set_user_bc_fns(self, left_bc=None, right_bc=None):
        if left_bc is not None:
            self.left_bc_fn = left_bc
        if right_bc is not None:
            self.right_bc_fn = right_bc

    def flatten(self):
        x_grids = []
        Q_grids = []
        leaves = self.get_leaves()
        for i, leaf in enumerate(leaves):
            slice_start = NUM_GHOST
            slice_end = -NUM_GHOST
            if i == 0:
                slice_start = None
            if i == len(leaves) - 1:
                slice_end = None

            x_grids.append(leaf.x_pos[slice_start:slice_end])
            Q_grids.append(leaf.Q[:, slice_start:slice_end])
        x_grids = np.concatenate(x_grids)
        Q_grids = np.concatenate(Q_grids, axis=1)
        return x_grids, Q_grids

    def get_dt_ch(self, max_cfl: float=0.6, gamma: float=DEFAULT_GAMMA):
        dts, chs = zip(*[b.get_dt_ch(max_cfl=max_cfl, gamma=gamma) for b in self.get_leaves()])
        return min(dts), max(chs)


def enforce_flux_consistency(leaves, fluxes):
    # NOTE(cmo): This is only correct because we don't subcycle in time for the fine grids
    for i in range(len(leaves)):
        leaf = leaves[i]
        if i > 0:
            left = leaves[i-1]
            if left.depth > leaf.depth:
                fluxes[i][:, NUM_GHOST] = fluxes[i-1][:, -(NUM_GHOST+1)]

        if i < len(leaves) - 1:
            right = leaves[i+1]
            if right.depth > leaf.depth:
                fluxes[i][:, -(NUM_GHOST+1)] = fluxes[i+1][:, NUM_GHOST]

def enforce_flux_consistency_vector(leaves, depths, fluxes):
    # NOTE(cmo): This is only correct because we don't subcycle in time for the fine grids
    for i in range(leaves.shape[2]):
        if i > 0:
            if depths[i-1] > depths[i]:
                fluxes[:, NUM_GHOST, i] = fluxes[:, -(NUM_GHOST+1), i-1]

        if i < leaves.shape[2] - 1:
            if depths[i+1] > depths[i]:
                fluxes[:, -(NUM_GHOST+1), i] = fluxes[:, NUM_GHOST, i+1]


def fill_ghosts(leaves: List[Block]):
    for i in range(len(leaves)):
        if i != 0:
            # do left ghosts
            block = leaves[i]
            prev_block = leaves[i-1]
            if block.depth == prev_block.depth:
                block.Q[:, :NUM_GHOST] = prev_block.Q[:, -2*NUM_GHOST:-NUM_GHOST]
            elif block.depth > prev_block.depth:
                # refine

                # NOTE(cmo): Round up
                half_ghost = (NUM_GHOST + 1) // 2

                # NOTE(cmo): Minimise reconstruction size
                parent = block.Q[:, NUM_GHOST-1:NUM_GHOST + half_ghost + 1]
                duL = parent[:, 1:-1] - parent[:, :-2]
                duR = parent[:, 2:] - parent[:, 1:-1]
                delta = np.zeros_like(parent)
                delta[:, 1:-1] = slope_limiter(duL, duR)
                block.Q[:, 0:NUM_GHOST:2] = parent[:, 1:-1] - 0.25 * delta[:, 1:-1]
                block.Q[:, 1:NUM_GHOST:2] = parent[:, 1:-1] + 0.25 * delta[:, 1:-1]

                # Old np.interp approach
                # prev_grid = prev_block.x_pos
                # this_grid = block.x_pos
                # for v in range(NUM_EQ):
                #     block.U[v, :NUM_GHOST] = np.interp(this_grid[:NUM_GHOST], prev_grid, prev_block.U[v, :])
            else:
                # coarsen
                block.Q[:, :NUM_GHOST] = 0.5 * prev_block.Q[:, -3 * NUM_GHOST:-NUM_GHOST].reshape(NUM_EQ, NUM_GHOST, 2).sum(axis=2)
        if i != len(leaves) - 1:
            # do right ghosts
            block = leaves[i]
            next_block = leaves[i+1]
            if block.depth == next_block.depth:
                block.Q[:, -NUM_GHOST:] = next_block.Q[:, NUM_GHOST:2*NUM_GHOST]
            elif block.depth > next_block.depth:
                # refine

                # NOTE(cmo): Round up
                half_ghost = (NUM_GHOST + 1) // 2

                # NOTE(cmo): Minimise reconstruction size
                parent = block.Q[:, -(NUM_GHOST + half_ghost + 1):-(NUM_GHOST-1)]
                duL = parent[:, 1:-1] - parent[:, :-2]
                duR = parent[:, 2:] - parent[:, 1:-1]
                delta = np.zeros_like(parent)
                delta[:, 1:-1] = slope_limiter(duL, duR)
                block.Q[:, -NUM_GHOST::2] = parent[:, 1:-1] - 0.25 * delta[:, 1:-1]
                block.Q[:, -(NUM_GHOST-1)::2] = parent[:, 1:-1] + 0.25 * delta[:, 1:-1]

                # Old np.interp approach
                # next_grid = next_block.x_pos
                # parent = block.x_pos
                # for v in range(NUM_EQ):
                #     block.U[v, -NUM_GHOST:] = np.interp(parent[-NUM_GHOST:], next_grid, next_block.U[v, :])
            else:
                # coarsen
                block.Q[:, -NUM_GHOST:] = 0.5 * next_block.Q[:, NUM_GHOST:3*NUM_GHOST].reshape(NUM_EQ, NUM_GHOST, 2).sum(axis=2)

def set_bcs(leaves: List[Block], dt, bc_modes, fixed_bc, left_bc_fn, right_bc_fn, gamma=DEFAULT_GAMMA):
    left = leaves[0]
    if bc_modes[0] == SYMMETRIC_BC:
        left.Q[:, :NUM_GHOST] = left.Q[:, NUM_GHOST:2*NUM_GHOST][:, ::-1]
    elif bc_modes[0] == REFLECTING_BC:
        left.Q[:, :NUM_GHOST] = left.Q[:, NUM_GHOST:2*NUM_GHOST][:, ::-1]
        left.Q[1, :NUM_GHOST] = -left.Q[1, NUM_GHOST:2*NUM_GHOST][::-1]
    elif bc_modes[0] == FIXED_BC:
        left.Q[:, :NUM_GHOST] = fixed_bc[0][:, None]
    elif bc_modes[0] == USER_BC:
        left_bc_fn(left, dt, gamma=gamma)

    right = leaves[-1]
    if bc_modes[1] == SYMMETRIC_BC:
        right.Q[:, -NUM_GHOST:] = right.Q[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]
    elif bc_modes[1] == REFLECTING_BC:
        right.Q[:, -NUM_GHOST:] = right.Q[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]
        right.Q[1, -NUM_GHOST:] = -right.Q[1, -2*NUM_GHOST:-NUM_GHOST][::-1]
    elif bc_modes[1] == FIXED_BC:
        right.Q[:, -NUM_GHOST:] = fixed_bc[1][:, None]
    elif bc_modes[1] == USER_BC:
        right_bc_fn(right, dt, gamma=gamma)

def rusanov_flux_with_padding(wL, wR, gamma: float=DEFAULT_GAMMA):
    # N.B. This takes the reconstructed faces in the cell frame, and returns a padded array with length (M+2*NUM_GHOST+1), i.e the flux at each interface in the block
    unpadded_flux = rusanov_flux(
        wR,
        np.roll(wL, -1, axis=1),
        gamma=gamma
    )
    if wL.ndim == 2:
        full_flux = np.empty((wL.shape[0], wL.shape[1]+1))
    else:
        full_flux = np.empty((wL.shape[0], wL.shape[1]+1, wL.shape[2]))
    full_flux[:, 1:] = unpadded_flux
    full_flux[:, :NUM_GHOST] = 0.0
    full_flux[:, -NUM_GHOST:] = 0.0
    return full_flux

@dataclass
class TimestepInfo:
    dt: float
    """timestep"""
    cfl: float
    """associated cfl"""
    c_h: float
    """max hyperbolic wave speed"""

def run_step(leaves: List[Block], ts: TimestepInfo, bc_modes, fixed_bcs, gamma: float=DEFAULT_GAMMA):
    scalar_grid = True
    Q_old = [l.Q.copy() for l in leaves]
    if not scalar_grid:
        Q_old_stack = np.stack(Q_old, axis=-1)
        sources = np.zeros_like(Q_old_stack)
    else:
        sources = [np.zeros_like(q) for q in Q_old]

    dt = ts.dt
    avg_mass = 1.0
    dt_scheme = [dt, 0.5 * dt]

    for substep, dt_sub in enumerate(dt_scheme):
        fluxes = []
        fill_ghosts(leaves)
        set_bcs(leaves, dt_sub, bc_modes, fixed_bcs, forest.left_bc_fn, forest.right_bc_fn, gamma=gamma)
        if scalar_grid:
            for s in sources:
                s[...] = 0.0
            for leaf, source in zip(leaves, sources):
                w = cons_to_prim(leaf.Q, gamma=gamma)
                # Relative to cells
                wL, wR = reconstruct_plm(w)
                # wL, wR = reconstruct_fog(w)
                interface_flux = rusanov_flux_with_padding(wL, wR, gamma=gamma)
                fluxes.append(interface_flux)

                if USE_CONDUCTION:
                    nh_tot = w[RHO] / (avg_mass * P_MASS)
                    y = 0.0
                    temperature = temperature_si(w[PRES], nh_tot, y)
                    compute_heatf_source(
                        temperature,
                        leaf.Q,
                        w,
                        source,
                        leaf.dx,
                        ts.dt,
                        ts.cfl,
                        ts.c_h,
                        gamma=gamma
                    )

            enforce_flux_consistency(leaves, fluxes)

            for idx, (block, source) in enumerate(zip(leaves, sources)):
                dx = block.dx
                flux = fluxes[idx]
                flux_div = flux[:, NUM_GHOST+1:-NUM_GHOST] - flux[:, NUM_GHOST:-(NUM_GHOST+1)]
                flux_update = - dt_sub / dx * flux_div + source[:, NUM_GHOST:-NUM_GHOST] * dt_sub
                if substep == 0:
                    block.Q[:, NUM_GHOST:-NUM_GHOST] += flux_update
                else:
                    block.Q[:, NUM_GHOST:-NUM_GHOST] = 0.5 * (
                        Q_old[idx][:, NUM_GHOST:-NUM_GHOST] + block.Q[:, NUM_GHOST:-NUM_GHOST]
                    ) + flux_update
        else:
            sources[...] = 0.0
            # NOTE(cmo): This isn't a super awesome layout for SIMD etc, but
            # it's the minimum changes for vectorising
            leaf_stack = np.stack([l.Q for l in leaves], axis=-1)
            depths = np.array([l.depth for l in leaves], dtype=np.int32)
            dxs = np.array([l.dx for l in leaves])
            w = cons_to_prim(leaf_stack, gamma=gamma)
            wL, wR = reconstruct_plm(w)
            flux_stack = rusanov_flux_with_padding(wL, wR, gamma=gamma)

            if USE_CONDUCTION:
                nh_tot = w[RHO] / (avg_mass * P_MASS)
                y = 0.0
                temperature = temperature_si(w[PRES], nh_tot, y)
                compute_heatf_source(
                    temperature,
                    leaf_stack,
                    w,
                    sources,
                    dxs[None, None, :],
                    ts.dt,
                    ts.cfl,
                    ts.c_h,
                    gamma=gamma
                )

            enforce_flux_consistency_vector(leaf_stack, depths, flux_stack)

            flux_div = flux_stack[:, NUM_GHOST+1:-NUM_GHOST, :] - flux_stack[:, NUM_GHOST:-(NUM_GHOST+1), :]
            flux_update = - dt_sub / dxs[None, None, :] * flux_div + sources[:, NUM_GHOST:-NUM_GHOST] * dt_sub
            if substep == 0:
                leaf_stack[:, NUM_GHOST:-NUM_GHOST, :] += flux_update
            else:
                leaf_stack[:, NUM_GHOST:-NUM_GHOST, :] = 0.5 * (
                    Q_old_stack[:, NUM_GHOST:-NUM_GHOST, :] + leaf_stack[:, NUM_GHOST:-NUM_GHOST]
                ) + flux_update
            # NOTE(cmo): This copy is gonna be sloooooow
            for i, block in enumerate(leaves):
                block.Q[...] = leaf_stack[:, :, i]


def sod_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.where(x < 0.5, 1.0, 0.125),
        np.zeros_like(x),
        np.where(x < 0.5, 1.0, 0.1),
    ])
    return prim_to_cons(w, gamma=gamma)

def sod_bcs():
    return [SYMMETRIC_BC, SYMMETRIC_BC]

def big_sod_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.where(x < 0.5, 1.0, 0.125),
        np.zeros_like(x),
        np.where(x < 0.5, 10.0, 0.1),
    ])
    return prim_to_cons(w, gamma=gamma)

def big_sod_bcs():
    return [SYMMETRIC_BC, SYMMETRIC_BC]

def woodward_collela_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.ones_like(x),
        np.zeros_like(x),
        np.where(x < 0.1, 1e3, np.where(x < 0.9, 0.01, 100.0)),
    ])
    return prim_to_cons(w, gamma=gamma)

def woodward_collela_bcs():
    return [REFLECTING_BC, REFLECTING_BC]

def navarro_hypertc_test_ics(x, gamma=DEFAULT_GAMMA):
    temperature = 0.1 + 0.9*x**5
    rho = np.ones_like(x)
    v = np.zeros_like(x)
    avg_mass = 1.0
    p = 1.0 * rho / (avg_mass * P_MASS) * k_B * temperature

    if not USE_CONDUCTION:
        raise ValueError("Need conduction")

    w = np.empty((NUM_EQ, x.shape[0]))
    w[RHO] = rho
    w[VEL] = v
    w[PRES] = p
    w[HEATF] = 0.0
    return prim_to_cons(w, gamma=gamma)

def navarro_hypertc_test_bcs():
    return [USER_BC, USER_BC]

def navarro_hypertc_test_left_bc(block, dt, gamma=DEFAULT_GAMMA):
    block.Q[RHO, :NUM_GHOST] = 1.0
    block.Q[MOM, :NUM_GHOST] = 0.0
    p = 1.0 / P_MASS * k_B * 0.1
    block.Q[ENE, :NUM_GHOST] = p / (gamma - 1.0)
    # block.Q[HEATF, :NUM_GHOST] = block.Q[HEATF, NUM_GHOST:2*NUM_GHOST][::-1]
    block.Q[HEATF, :NUM_GHOST] = block.Q[HEATF, NUM_GHOST]

def navarro_hypertc_test_right_bc(block, dt, gamma=DEFAULT_GAMMA):
    block.Q[RHO, -NUM_GHOST:] = 1.0
    block.Q[MOM, -NUM_GHOST:] = 0.0
    p = 1.0 / P_MASS * k_B * 1.0
    block.Q[ENE, -NUM_GHOST:] = p / (gamma - 1.0)
    # block.Q[HEATF, -NUM_GHOST:] = block.Q[HEATF, -2*NUM_GHOST:-NUM_GHOST][::-1]
    block.Q[HEATF, -NUM_GHOST:] = block.Q[HEATF, -NUM_GHOST-1]

def run_sim(forest: Forest, bc_modes, max_time, max_cfl=0.5, max_steps=10_000_000, output_cadence=0.25):
    current_time = 0.0
    dt, max_ch = forest.get_dt_ch(max_cfl=max_cfl, gamma=DEFAULT_GAMMA)
    leaves = forest.get_leaves()
    snaps = []
    next_output = current_time + output_cadence
    snaps.append((current_time, *forest.flatten()))
    for i in range(max_steps):
        timestep_info = TimestepInfo(dt, max_cfl, max_ch)
        run_step(leaves, timestep_info, bc_modes, forest.fixed_bcs, gamma=DEFAULT_GAMMA)

        current_time += dt
        if current_time >= next_output:
            snaps.append((current_time, *forest.flatten()))
            next_output = current_time + output_cadence
        if i % 50 == 0 or current_time >= max_time:
            print(f"t: {current_time:.4f} s, dt: {dt:.2e} s, iter: {i:9d}, grids: {len(leaves):d}")
        if current_time >= max_time:
            break

        if i > 0 and i % REGRID_FREQ == 0:
            forest.regrid(refine_threshold=ERR_THRESH, deref_ratio=DEREF_RATIO)
            leaves = forest.get_leaves()
            set_bcs(leaves, dt, bc_modes, forest.fixed_bcs, forest.left_bc_fn, forest.right_bc_fn, gamma=DEFAULT_GAMMA)
            fill_ghosts(leaves)

        dt, max_ch = forest.get_dt_ch(max_cfl=max_cfl, gamma=DEFAULT_GAMMA)
        # dt = 1e-4
        # max_ch = min([leaf.dx for leaf in forest.get_leaves()]) / dt / max_cfl
        if current_time + dt > next_output:
            dt = next_output - current_time
            while current_time + dt < next_output:
                dt = np.nextafter(dt, np.inf)
    return snaps

def cons_to_temperature(Q, gamma=DEFAULT_GAMMA):
    w = cons_to_prim(Q, gamma=gamma)
    nh_tot = w[RHO] / (P_MASS)
    return temperature_si(w[PRES], nh_tot, 0.0)

if __name__ == '__main__':
    fixed_bcs = None
    user_bcs = None

    # ics = sod_ics
    # bcs = sod_bcs
    # max_time = 0.2

    # ics = big_sod_ics
    # bcs = big_sod_bcs
    # max_time = 0.1

    # ics = woodward_collela_ics
    # bcs = woodward_collela_bcs
    # max_time = 0.038

    # k_B = 1.0
    # P_MASS = 1.0
    P_MASS = k_B
    ics = navarro_hypertc_test_ics
    bcs = navarro_hypertc_test_bcs
    user_bcs = (navarro_hypertc_test_left_bc, navarro_hypertc_test_right_bc)
    max_time = 1.0

    forest = Forest(0.0, 1.0, N=1, M=250, L=0)
    forest.set_ics(ics)
    if fixed_bcs is not None:
        forest.set_fixed_bcs(fixed_bcs)
    if user_bcs is not None:
        forest.set_user_bc_fns(left_bc=user_bcs[0], right_bc=user_bcs[1])
    # forest_uni = Forest(0.0, 1.0, N=1, M=2048, L=0)
    # forest_uni.set_ics(ics)
    bc_modes = bcs()

    x0, Q0 = forest.flatten()
    states = run_sim(forest, bc_modes, max_time=max_time)
    # run_sim(forest_uni, bc_modes, max_time=max_time)

    flat_x, flat_Q = forest.flatten()
    # flat_x_u, flat_Q_u = forest_uni.flatten()
