import astropy.constants as const
# Sim
# DEFAULT_GAMMA = 1.4
DEFAULT_GAMMA = 5/3
# DEFAULT_GAMMA = 2.0
USE_CONDUCTION = True
CONDUCTION_ONLY = True
NUM_GHOST = 2
# Base particle mass, combined with eos mean mass [kg]
P_MASS = 1.6737830080950003e-27
MEAN_MOLECULAR_MASS = 0.5 # Pure H plasma

# AMR
ERR_THRESH = 0.1
DEREF_RATIO = 0.125
REGRID_FREQ = 5

# BC Idxs
SYMMETRIC_BC = 0
REFLECTING_BC = 1
FIXED_BC = 2
USER_BC = 3

# Idxs
RHO = 0
MOM = 1
ENE = 2

VEL = 1
PRES = 2

HEATF = 3 if USE_CONDUCTION else 1024

# Conduction
COND_HTC = 0
COND_IMPLICIT = 1
COND_MODE = COND_HTC
HYPERTC_IN_FLUX_VECTOR = True
SATURATE_HEAT_FLUX = False
SPITZER_CONDUCTIVITY = True
HTC_HYPERDIFFUSION = 5e-2
KAPPA0 = 1.0
# KAPPA0 = 8e-12 # Hyperbolic conduction coefficient [W m-1 K-7/2]. Default value is Spitzer like (i.e. (8e-7 erg cm-1 s-1 K-7/2)).

# Constants
k_B = const.k_B.value


NUM_EQ = 3 + (USE_CONDUCTION if COND_MODE == COND_HTC else 0)