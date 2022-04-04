import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
import scipy.optimize

from He4_data import DISTANCE, PAIR_DISTRIBUTION
from He4_data import RESOLUTION_SIGMA, RHO

import warnings

g = scipy.interpolate.interp1d(
	DISTANCE,
	PAIR_DISTRIBUTION,
	kind='cubic'
)

g.__doc__ = """
	Cubic interpolation of the pair distribution function.
"""

# For V(r), I use the Aziz HFD-B3-FCI1 He-He potential.
# Reference: R.A. Aziz et al. Phys. Rev. Lett. 74, 1585 (1995).

# First, I define the coefficients of the potential.
# For the potential in Kelvins (K), use eps = 10.94.  Here it has been
# converted to meV using: 1 meV = 11.6 K.

ASTAR = 192215.29
ALPHASTAR = 10.73520708
C6 = 1.34920045
C8 = 0.41365922
C10 = 0.17078164
BETASTAR = -1.89296514
RM = 2.97
DAZIZ = 1.4132
EPS = 10.94/11.6

# Second, I define the Aziz potential itself, using their notation.

def f(x):
    if x < 1.241314:
        return np.exp(-(DAZIZ/x - 1.0)**2)
    else:
        return 1.0

def vstar(x):
    return ASTAR * np.exp(-ALPHASTAR*x + BETASTAR*x**2)\
	- f(x) * (C6/x**6 + C8/x**8 + C10/x**10)

def v(r):
    """
    Aziz HFD-B3-FCI1 He-He potential

	Parameters
	----------
	r : float
		Interparticle spacing.

	Returns
	-------
	float
		Potential in units of meV.
    """
    return EPS * vstar(r/RM)


# Calculation of Classical Turning Point

def v_prime(r, q):
    return (4.0/1.04436)*(1.0/q**2)*v(r)

def turning_point(r0, b, q):
    return 1.0 - v_prime(r0,q) - b**2/r0**2

def r0(b,q):
    """
    Classical Turning point.
    
    Parameters
    ----------
    b : float
        Impact parameter in units of \AA.
    q : float
        Wavevector transfer in units of \AA^{-1}

    Returns
    -------
    float
        Classical turning point in units of \AA.

    """
    return scipy.optimize.brentq(turning_point, 0.1, 4, args=(b,q))


def quad_no_warnings(*args, **kwargs):
	with warnings.catch_warnings():
		# IntegrationWarning from scipy.integrate has been suppressed.  The
        # method produces correct numerical values for the integral, but it 
        # yields warnings about the uncertainty.  These warnings have been
        # muted because these uncertainties are nowhere needed in the program.
		warnings.simplefilter(
            action='ignore', category=scipy.integrate.IntegrationWarning)
		return scipy.integrate.quad(*args, **kwargs)

#
# Calculation of Jeffreys-Wentzel-Kramers-Brillouin phase shifts.  See
# equation (38) of first paper listed in the header.  I only retain the
# negative phase shifts due to the replusive part of the potential.
#

def phase_integrand(rprime, b, q):
    return np.sqrt(turning_point(rprime, b, q)) - 1.0


def delta_exact(b, q):
	local_r0 = r0(b,q)
	local_int = quad_no_warnings(
        phase_integrand, local_r0, np.inf, args=(b,q))[0]

	return 0.5 * q * (0.5 * b * scipy.pi - local_r0 + local_int)


def phase_shift(b, q):
    local_delta = delta_exact(b, q)
    if local_delta <= 0.0:
        return local_delta
    else:
        return 0.0

def fb(b, q):
	"""
	Complex phase fb of recoiling helium atoms.

	Parameters
	----------
	b : float
		Impact parameter in units of \AA.

	q : float
		Wavevector transfer in units of \AA^{-1}.

	Returns
	-------
	complex
		Phase fb.
	"""
	return np.expm1((0.0 + 2.0j) * phase_shift(b,q ))


# Calculation of Gamma(x).
# First, I define the functions in the asymptotic limit (infinitely hard
# spheres at Q = inf).

def integrand_asymp(b, x):
    return b * g(np.sqrt(b**2 + x**2))[()]

def gamma_asymp(x, R):
	return 2.0*scipy.pi*(0.0 + 1.0j)*RHO*quad_no_warnings(integrand_asymp,
                                                           0, R, args = (x))[0]
# The hard sphere radius R is adjustable.

# Second, I define the relevant functions for the Aziz potential.
# The complex integrand is broken up into real and complex parts.  These are
# separately integrated over and then recombined to yield the total function.

def integrand_hcpt(b, x, q):
	return fb(b, q) * integrand_asymp(b, x)

def real_integrand_hcpt(b, x, q):
    return np.real(integrand_hcpt(b,x,q)[()])

def imag_integrand_hcpt(b, x, q):
    return np.imag(integrand_hcpt(b,x,q)[()])

def real_gamma_hcpt(x, q):
    return quad_no_warnings(real_integrand_hcpt, 0,\
	 4, args = (x, q))[0]

def imag_gamma_hcpt(x, q):
    return quad_no_warnings(imag_integrand_hcpt, 0,\
	 4, args = (x, q))[0]

def gamma_hcpt(x, q):
    """
    HCPT Gamma(x).

    Parameters
    ----------
    x : float
        Distance in units of \AA.
    q : float
        Wavevector transfer in units of \AA^{-1}.

    Returns
    -------
    complex
        Gamma(x).

    """
    return 2.0*scipy.pi*RHO*(0.0 - 1.0j)*(
        (1.0 + 0.0j) * real_gamma_hcpt(x, q) +
        (0.0 + 1.0j) * imag_gamma_hcpt(x, q))

# Integrals over gamma for HCPT

def over_gamma(x_prime, gamma_part):
        return quad_no_warnings(gamma_part, 0.0, x_prime)[0]

def R(s, gamma_r, gamma_i):
    """
    Final state effect function R(s, Q).

    Parameters
    ----------
    s : float
        Distance in units of \AA.
    gamma_r : function
        The real part of Gamma(x).  This is obtained by a scipy interpolation.
    gamma_i : function
        The imaginary part of Gamma(x).  This is obtained by a scipy 
        interpolation.

    Returns
    -------
    complex

    """
    return np.exp((0.0 + 1.0j)*(
        1.0*over_gamma(s, gamma_r) + 1.0j*over_gamma(s, gamma_i)))


# The following two functions are used to represent the Path Integral Monte
# Carlo calculations in a smooth manner.  The absolute temperature is 1.09 K.

def Gaussian(x, area, center, FWHM):
    
    """
	Gaussian peak.

	Parameters
	----------
	x : float
		co-ordinate
        
    area : float
        Integrated intensity of the peak.
        
    center : float
        Position of the peak.  Equivalent to the first moment.
        
    FWHM : float
        Full-width at half-maximum of the peak.
        The second moment, sigma, is equal to FWHM/2.3548.

	Returns
	-------
	float
		Peak intensity/amplitude at position x.
	"""
    
    prefactor = (area/np.sqrt(2*np.pi*(FWHM/2.3548)**2))
    peak = np.exp(-0.5*(x-center)**2/(FWHM/2.3548)**2)

    return prefactor*peak

def OBDM(s):
    """
    One-body density matrix of liquid 4-He at 1.09 K.

    Parameters
    ----------
    s : float
        Distance in units of \AA.

    Returns
    -------
    float

    """
    
    return 7.357e-2 + Gaussian(s, 2.271, 0, 3.189) + \
        Gaussian(s, 5.227e-1, 0, 1.921) + \
            Gaussian(s, -9.217e-3, 4.098, 1.307) + \
                Gaussian(s, 4.603e-3, 6.654, 1.099) + \
                    Gaussian(s, -4.179e-3, 7.614, 2.502) +\
                        Gaussian(s, 1.504e-3, 9.646, 8.413e-1)


def resolution(x):
	return np.exp(-0.5*x**2*RESOLUTION_SIGMA**2)


# This function expresses the theoretical predictions in Fourier time.  The
# complete function is a product of the one-body density matrix from Path
# Integral Monte Carlo, the final state effect function from Hard Core
# Perturbation Theory, and the instrumental resolution.

def prediction(s, gamma_r, gamma_i):
    return OBDM(s)*R(s, gamma_r, gamma_i)*resolution(s)


# These functions carry out the Fourier cosine transform of the theoretical
# predictions, allowing a direct comparison with experiment.

def pred_FT_integrand(x, Y, gamma_r, gamma_i):
    return np.real(np.exp((0.0 + 1.0j)*Y*x) * prediction(x, gamma_r, gamma_i))

def theory(Y, gamma_r, gamma_i):
    """
    Theoretical prediction for the neutron Compton profile.

    Parameters
    ----------
    Y : float
        West scaling variable Y.
    gamma_r : function
        The real part of Gamma(x).  This is obtained by a scipy interpolation.
    gamma_i : TYPE
        The imaginary part of Gamma(x).  This is obtained by a scipy 
        interpolation.

    Returns
    -------
    float
        J(Y, Q) in units of \AA^{-1}

    """
    
    factor = (1.0/np.pi)
    transform = quad_no_warnings(
        pred_FT_integrand, 0, 9.5, args = (Y, gamma_r, gamma_i))[0]
    return factor*transform
