import astropy.constants as const
# Sim
DEFAULT_GAMMA = 1.4
USE_CONDUCTION = True
CONDUCTION_ONLY = True
NUM_EQ = 3 + USE_CONDUCTION
NUM_GHOST = 2
# Base particle mass, combined with eos mean mass [kg]
P_MASS = 1.6737830080950003e-27

# AMR
ERR_THRESH = 0.1
DEREF_RATIO = 0.125
REGRID_FREQ = 5

# BC Idxs
SYMMETRIC_BC = 0
REFLECTING_BC = 1
FIXED_BC = 2

# Idxs
RHO = 0
MOM = 1
ENE = 2

VEL = 1
PRES = 2

HEATF = 3 if USE_CONDUCTION else 1024

# Conduction
HYPERTC_IN_FLUX_VECTOR = True
SATURATE_HEAT_FLUX = True
SPITZER_CONDUCTIVITY = True
HYPERTC_KAPPA = 100000.0
# HYPERTC_KAPPA = 8e-12 # Hyperbolic conduction coefficient [W m-1 K-7/2]. Default value is Spitzer like (i.e. (8e-7 erg cm-1 s-1 K-7/2)).

# Constants
k_B = const.k_B.value
