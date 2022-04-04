# Name: He4_function_test_script.py
# Purpose: This program is intended to verify that code used to generate the
# Hard Core Perturbation Theory predictions produces correct results.
#
# Author: Timothy R. Prisk
# Contact: tprisk@alumni.iu.edu
#


# Import needed modules.
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from He4_functions import v, g, r0, phase_shift, OBDM, R, resolution
from He4_functions import prediction
from He4_data import Q, DISTANCE, PAIR_DISTRIBUTION


# Verify that the shape of the interatomic potential is correct.

r_test = np.arange(2.6, 4.3, 0.05)
v_test = np.zeros(len(r_test))

for j in range(len(r_test)):
    v_test[j] = 11.6*v(r_test[j]) # 1 meV = 11.6 K

plt.plot(r_test, v_test, 'k-')
plt.xlabel('r [$\AA$]')
plt.ylabel('V(r) [K]')
plt.title('Aziz Potential')
plt.show()


# Verify that the interpolation of g(r) is correct.

r_structure_test = np.arange(0.0, 15.0, 0.01)
g_structure_test = np.zeros(len(r_structure_test))

for k in range(len(r_structure_test)):
    g_structure_test[k] = g(r_structure_test[k])

plt.plot(DISTANCE, PAIR_DISTRIBUTION, 'ko', label = 'Exp')
plt.plot(r_structure_test, g_structure_test, 'r-', label = 'interpolation')
plt.xlabel('r [$\AA$]')
plt.xlim([0.0, 15.0])
plt.ylabel('g(r) [K]')
plt.title('Pair Distribution Function')
plt.show()


# Verify turning point calculation is correct.  Use an impact parameter of
# of zero (b = 0).

BTEST = 0.0

q_test = np.arange(5.0, 220.0, 10.0)
r0_test = np.zeros(len(q_test))

for k in range(len(q_test)):
    r0_test[k] = r0(0.0, q_test[k])

plt.semilogx(q_test, r0_test, 'k-')
plt.xlabel('Q[$\AA^{-1}$]')
plt.ylabel("$r_0$ ($\AA$)")
plt.title('Classical Turning Point')
plt.show()


# Check the JWKB phase shift calculations.

b_test = np.arange(0.0, 4.0, 0.05)
phase_test = np.zeros(len(b_test))

for k in range(len(phase_test)):
    phase_test[k] = phase_shift(b_test[k], Q)

plt.plot(b_test, phase_test, 'k-')
plt.xlabel('Impact parameter $b$($\AA$)')
plt.ylabel('$\delta_b$')
plt.title('JKWB phase shift')
plt.show()

# Visualize the HCPT gamma function.  Pre-existing files are loaded because
# the determination of Gamma is computation intensive.

x_test = np.load("x_27.0.npy")
rGamma_test = np.load("rGammas_27.0.npy")
iGamma_test = np.load("iGammas_27.0.npy")

plt.plot(x_test, rGamma_test, 'k--', label = '$Re[\Gamma(x)]$')
plt.plot(x_test, iGamma_test, 'k-', label = '$Im[\Gamma(x)]$')
plt.xlabel('x ($\AA$)')
plt.legend(loc = 'upper right')
plt.show()

# Interpolation over real and imaginary parts of gamma.
gamma_r = interpolate.interp1d(x_test, rGamma_test, kind = 'cubic')
gamma_i = interpolate.interp1d(x_test, iGamma_test, kind = 'cubic')

# Verify that the one-body density matrix, final state effect function,
# resolution, and final predictions look reasonable.

s_test = np.arange(0.0, 15.0, 0.1)

obdm_test = np.zeros(len(s_test), dtype = float)

r_test_full = np.zeros(len(s_test), dtype = complex)
r_test_reals = np.zeros(len(s_test), dtype = float)
r_test_imags = np.zeros(len(s_test), dtype = float)

resolution_test = np.zeros(len(s_test), dtype = float)

prediction_test = np.zeros(len(s_test), dtype = complex)
prediction_test_reals = np.zeros(len(s_test), dtype = float)
prediction_test_imags = np.zeros(len(s_test), dtype = float)

for k in range(len(s_test)):
    obdm_test[k] = OBDM(s_test[k])
    r_test_full[k] = R(s_test[k], gamma_r, gamma_i)
    resolution_test[k] = resolution(s_test[k])
    prediction_test[k] = prediction(s_test[k], gamma_r, gamma_i)

r_test_reals = np.real(r_test_full)
r_test_imags = np.imag(r_test_full)

prediction_test_reals = np.real(prediction_test)
prediction_test_imags = np.imag(prediction_test)

plt.plot(s_test, obdm_test, 'k-', label = 'OBDM')
plt.plot(s_test, r_test_reals, 'r-', label = 'Re HCPT')
plt.plot(s_test, r_test_imags, 'r--', label = 'Im HCPT')
plt.plot(s_test, resolution_test, 'b-', label = 'Res.')
plt.plot(s_test, prediction_test_reals, 'g-', label = 'Re J(s, Q)')
plt.plot(s_test, prediction_test_imags, 'g--', label = 'Im J(s, Q)')
plt.xlabel('s ($\AA$)')
plt.legend(loc = 'upper right')
plt.show()




