from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

import numpy as np
from euler import DEFAULT_GAMMA, cons_to_prim, prim_to_cons, prim_to_flux, rusanov_flux, sound_speed
from reconstruct import reconstruct_fog, reconstruct_plm, slope_limiter

NUM_EQ = 3
NUM_GHOST = 2

ERR_THRESH = 0.2
DEREF_RATIO = 0.125
REGRID_FREQ = 5

SYMMETRIC_BC = 0
REFLECTING_BC = 1

class Block:
    def __init__(self, depth, max_depth, M, x0, x1, parent=None):
        self.depth = depth
        self.max_depth = max_depth
        self.M = M
        self.x0 = x0
        self.x1 = x1
        self.dx = (x1 - x0) / M

        self.U = np.zeros((NUM_EQ, M + 2 * NUM_GHOST))

        # Child blocks (None if leaf)
        self.left: Optional[Block] = None
        self.right: Optional[Block] = None
        self.parent: Optional[Block] = parent

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def x_pos(self):
        return (self.x0 - NUM_GHOST * self.dx) + np.arange(self.U.shape[1]) * self.dx

    def refine(self):
        if self.depth == self.max_depth:
            return

        xm = 0.5 * (self.x0 + self.x1)

        self.left = Block(self.depth+1, self.max_depth, self.M, self.x0, xm, parent=self)
        self.right = Block(self.depth+1, self.max_depth, self.M, xm, self.x1, parent=self)

        half_grid = self.U.shape[1] // 2

        # Prolongation -- ghosts aren't handled correctly, but that's fine, they
        # get filled before each step and also before CFL calculation
        duL = self.U[:, 1:-1] - self.U[:, :-2]
        duR = self.U[:, 2:] - self.U[:, 1:-1]
        delta = np.zeros_like(self.U)
        delta[:, 1:-1] = slope_limiter(duL, duR)

        self.left.U[:, NUM_GHOST:-NUM_GHOST:2] = (
            self.U[:, NUM_GHOST:half_grid] - 0.25 * delta[:, NUM_GHOST:half_grid]
        )
        self.left.U[:, NUM_GHOST+1:-NUM_GHOST:2] = (
            self.U[:, NUM_GHOST:half_grid] + 0.25 * delta[:, NUM_GHOST:half_grid]
        )

        self.right.U[:, NUM_GHOST:-NUM_GHOST:2] = (
            self.U[:, half_grid:-NUM_GHOST] - 0.25 * delta[:, half_grid:-NUM_GHOST]
        )
        self.right.U[:, NUM_GHOST+1:-NUM_GHOST:2] = (
            self.U[:, half_grid:-NUM_GHOST] + 0.25 * delta[:, half_grid:-NUM_GHOST]
        )

    def coarsen(self):
        if self.is_leaf:
            return

        # Simplest restriction -- ignore ghosts. Rebuild grid and local average
        self.U[:, NUM_GHOST:-NUM_GHOST] = 0.5 * np.concat(
            [
                self.left.U[:, NUM_GHOST:-NUM_GHOST],
                self.right.U[:, NUM_GHOST:-NUM_GHOST],
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
        rho = self.U[0]

        d2 = np.abs(rho[:-2] - 2*rho[1:-1] + rho[2:])
        d1 = np.abs(rho[2:] - rho[1:-1]) + np.abs(rho[1:-1] - rho[:-2])
        return np.max(d2 / (d1 + eps))

    def get_dt(self, max_cfl: float = 0.6, gamma: float=DEFAULT_GAMMA):
        w = cons_to_prim(self.U)
        cs = sound_speed(w, gamma=gamma)
        fast_speed = np.abs(w[1]) + cs

        dt_local = max_cfl * self.dx / fast_speed
        return np.min(dt_local)

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
                leaf.U[:] = ics(leaf.x_pos, gamma=gamma)

            self.regrid(refine_threshold=ERR_THRESH, deref_ratio=DEREF_RATIO)

    def flatten(self):
        x_grids = []
        U_grids = []
        for leaf in self.get_leaves():
            x_grids.append(leaf.x_pos[NUM_GHOST:-NUM_GHOST])
            U_grids.append(leaf.U[:, NUM_GHOST:-NUM_GHOST])
        x_grids = np.concatenate(x_grids)
        U_grids = np.concatenate(U_grids, axis=1)
        return x_grids, U_grids

    def get_dt(self, max_cfl: float=0.6, gamma: float=DEFAULT_GAMMA):
        dts = [b.get_dt(max_cfl=max_cfl, gamma=gamma) for b in self.get_leaves()]
        return min(dts)


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


def fill_ghosts(leaves: List[Block]):
    for i in range(len(leaves)):
        if i != 0:
            # do left ghosts
            block = leaves[i]
            prev_block = leaves[i-1]
            if block.depth == prev_block.depth:
                block.U[:, :NUM_GHOST] = prev_block.U[:, -2*NUM_GHOST:-NUM_GHOST]
            elif block.depth > prev_block.depth:
                # refine

                # NOTE(cmo): Round up
                half_ghost = (NUM_GHOST + 1) // 2

                # NOTE(cmo): Minimise reconstruction size
                parent = block.U[:, NUM_GHOST-1:NUM_GHOST + half_ghost + 1]
                duL = parent[:, 1:-1] - parent[:, :-2]
                duR = parent[:, 2:] - parent[:, 1:-1]
                delta = np.zeros_like(parent)
                delta[:, 1:-1] = slope_limiter(duL, duR)
                block.U[:, 0:NUM_GHOST:2] = parent[:, 1:-1] - 0.25 * delta[:, 1:-1]
                block.U[:, 1:NUM_GHOST:2] = parent[:, 1:-1] + 0.25 * delta[:, 1:-1]

                # Old np.interp approach
                # prev_grid = prev_block.x_pos
                # this_grid = block.x_pos
                # for v in range(NUM_EQ):
                #     block.U[v, :NUM_GHOST] = np.interp(this_grid[:NUM_GHOST], prev_grid, prev_block.U[v, :])
            else:
                # coarsen
                block.U[:, :NUM_GHOST] = 0.5 * prev_block.U[:, -3 * NUM_GHOST:-NUM_GHOST].reshape(NUM_EQ, NUM_GHOST, 2).sum(axis=2)
        if i != len(leaves) - 1:
            # do right ghosts
            block = leaves[i]
            next_block = leaves[i+1]
            if block.depth == next_block.depth:
                block.U[:, -NUM_GHOST:] = next_block.U[:, NUM_GHOST:2*NUM_GHOST]
            elif block.depth > next_block.depth:
                # refine

                # NOTE(cmo): Round up
                half_ghost = (NUM_GHOST + 1) // 2

                # NOTE(cmo): Minimise reconstruction size
                parent = block.U[:, -(NUM_GHOST + half_ghost + 1):-(NUM_GHOST-1)]
                duL = parent[:, 1:-1] - parent[:, :-2]
                duR = parent[:, 2:] - parent[:, 1:-1]
                delta = np.zeros_like(parent)
                delta[:, 1:-1] = slope_limiter(duL, duR)
                block.U[:, -NUM_GHOST::2] = parent[:, 1:-1] - 0.25 * delta[:, 1:-1]
                block.U[:, -(NUM_GHOST-1)::2] = parent[:, 1:-1] + 0.25 * delta[:, 1:-1]

                # Old np.interp approach
                # next_grid = next_block.x_pos
                # parent = block.x_pos
                # for v in range(NUM_EQ):
                #     block.U[v, -NUM_GHOST:] = np.interp(parent[-NUM_GHOST:], next_grid, next_block.U[v, :])
            else:
                # coarsen
                block.U[:, -NUM_GHOST:] = 0.5 * next_block.U[:, NUM_GHOST:3*NUM_GHOST].reshape(NUM_EQ, NUM_GHOST, 2).sum(axis=2)

def set_bcs(leaves: List[Block], dt, bc_modes):
    left = leaves[0]
    if bc_modes[0] == SYMMETRIC_BC:
        left.U[:, :NUM_GHOST] = left.U[:, NUM_GHOST:2*NUM_GHOST][:, ::-1]
    elif bc_modes[0] == REFLECTING_BC:
        left.U[:, :NUM_GHOST] = left.U[:, NUM_GHOST:2*NUM_GHOST][:, ::-1]
        left.U[1, :NUM_GHOST] = -left.U[1, NUM_GHOST:2*NUM_GHOST][::-1]

    right = leaves[-1]
    if bc_modes[1] == SYMMETRIC_BC:
        right.U[:, -NUM_GHOST:] = right.U[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]
    elif bc_modes[1] == REFLECTING_BC:
        right.U[:, -NUM_GHOST:] = right.U[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]
        right.U[1, -NUM_GHOST:] = -right.U[1, -2*NUM_GHOST:-NUM_GHOST][::-1]

def rusanov_flux_with_padding(wL, wR, gamma: float=DEFAULT_GAMMA):
    # N.B. This takes the reconstructed faces in the cell frame, and returns a padded array with length (M+2*NUM_GHOST+1), i.e the flux at each interface in the block
    unpadded_flux = rusanov_flux(
        wR,
        np.roll(wL, -1, axis=1),
        gamma=gamma
    )
    full_flux = np.empty((wL.shape[0], wL.shape[1]+1))
    full_flux[:, 1:] = unpadded_flux
    full_flux[:, :NUM_GHOST] = 0.0
    full_flux[:, -NUM_GHOST:] = 0.0
    return full_flux

def run_step(leaves, dt, bc_modes, gamma: float=DEFAULT_GAMMA):
    U_old = [l.U.copy() for l in leaves]

    dt_scheme = [dt, 0.5 * dt]

    for substep, dt_sub in enumerate(dt_scheme):
        fluxes = []
        fill_ghosts(leaves)
        set_bcs(leaves, dt_sub, bc_modes)
        for leaf in leaves:
            w = cons_to_prim(leaf.U, gamma=gamma)
            # Relative to cells
            wL, wR = reconstruct_plm(w)
            # wL, wR = reconstruct_fog(w)
            interface_flux = rusanov_flux_with_padding(wL, wR, gamma=gamma)
            fluxes.append(interface_flux)

        enforce_flux_consistency(leaves, fluxes)

        for idx, block in enumerate(leaves):
            dx = block.dx
            flux = fluxes[idx]
            flux_div = flux[:, NUM_GHOST+1:-NUM_GHOST] - flux[:, NUM_GHOST:-(NUM_GHOST+1)]
            # TODO(cmo): Source terms
            flux_update = - dt_sub / dx * flux_div
            if substep == 0:
                block.U[:, NUM_GHOST:-NUM_GHOST] += flux_update
            else:
                block.U[:, NUM_GHOST:-NUM_GHOST] = 0.5 * (
                    U_old[idx][:, NUM_GHOST:-NUM_GHOST] + block.U[:, NUM_GHOST:-NUM_GHOST]
                ) + flux_update

def sod_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.where(x < 0.5, 1.0, 0.125),
        np.zeros_like(x),
        np.where(x < 0.5, 1.0, 0.1),
    ])
    return prim_to_cons(w, gamma=gamma)

def sod_bcs():
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

def run_sim(forest, bc_modes, max_time, max_cfl=0.5, max_steps=10_000_000):
    current_time = 0.0
    dt = forest.get_dt(max_cfl=max_cfl, gamma=DEFAULT_GAMMA)
    leaves = forest.get_leaves()
    for i in range(max_steps):
        run_step(leaves, dt, bc_modes, gamma=DEFAULT_GAMMA)

        current_time += dt
        if current_time >= max_time:
            break

        if i > 0 and i % REGRID_FREQ == 0:
            forest.regrid(refine_threshold=ERR_THRESH, deref_ratio=DEREF_RATIO)
            leaves = forest.get_leaves()
            set_bcs(leaves, dt, bc_modes)
            fill_ghosts(leaves)

        dt = forest.get_dt(max_cfl=max_cfl, gamma=DEFAULT_GAMMA)
        if current_time + dt > max_time:
            dt = max_time - current_time
            while current_time + dt < max_time:
                dt = np.nextafter(dt, np.inf)
        if i % 50 == 0:
            print(f"t: {current_time:.4f} s, dt: {dt:.2e} s, iter: {i:9d}, grids: {len(leaves):d}")

if __name__ == '__main__':
    ics = sod_ics
    bcs = sod_bcs
    max_time = 0.2

    # ics = woodward_collela_ics
    # bcs = woodward_collela_bcs
    # max_time = 0.038

    forest = Forest(0.0, 1.0, N=8, M=16, L=3)
    forest.set_ics(ics)
    forest_uni = Forest(0.0, 1.0, N=1, M=1024, L=0)
    forest_uni.set_ics(ics)
    bc_modes = bcs()

    run_sim(forest, bc_modes, max_time=max_time)
    run_sim(forest_uni, bc_modes, max_time=max_time)

    flat_x, flat_U = forest.flatten()
    flat_x_u, flat_U_u = forest_uni.flatten()
