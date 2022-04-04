# Name: He4_calc.py
# Purpose: This program calculates the neutron Compton profile of superfluid
# helium and then compares the result of the calculation with experiment.
#
# Author: Timothy R. Prisk
# Contact: tprisk@alumni.iu.edu
#
# Notes: This program calculates the final state effect broadening
# function R(Y, Q) of liquid 4He using Silver's Hard Core
# Perturbation Theory (HCPT).  The theory is described in print here:
# R.N. Silver, Phys. Rev. B 37 3794 (1988).
# R.N. Silver, Phys Rev. B 38 2283 (1988)
# R.N. Silver, Phys. Rev. B 39, 4022 (1989).
#
# Variables and functions in the program are named in order to match as 
# closely as possible the choice of notation in Silver's papers.
#


import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from He4_functions import gamma_hcpt, theory
from He4_data import Q, Y_data, J_data, err_data


def main():
    
    # Multicore calculation of Gamma(x) and J(Y, Q).
    # First, I generate a "table" of Gamma(x).  Then, I interpolate over the 
    # table in order to obtain smooth functions for its real and imaginary
    # parts.  Finally, a theoretical prediction for J(Y, Q) is generated.

    x_vals = np.arange(0.0, 30.0, 0.5)

    with mp.Pool() as pool:
        gamma_vals = pool.starmap(gamma_hcpt, ((x, Q) for x in x_vals))

        gamma_r = interpolate.interp1d(
            x_vals, np.real(gamma_vals), kind = 'cubic')
        gamma_i = interpolate.interp1d(
            x_vals, np.imag(gamma_vals), kind = 'cubic')

        Y_theory = np.arange(-4.0, 4.0, 0.25)
        J_theory = pool.starmap(
            theory, ((Y, gamma_r, gamma_i) for Y in Y_theory))

    
    # Save the results.
    gamma_vals_real = np.real(gamma_vals)
    gamma_vals_imag = np.imag(gamma_vals)
       
    np.savetxt("x_vals.txt", x_vals)
    np.savetxt("gamma_vals_real.txt", gamma_vals_real)
    np.savetxt("gamma_vals_imag.txt", gamma_vals_imag)

    np.savetxt("Y_theory.txt", Y_theory)
    np.savetxt("J_theory.txt", J_theory)
    
    np.savetxt("Y_data.txt", Y_data)
    np.savetxt("J_data.txt", J_data)
    np.savetxt("err_data.txt", err_data)
    
    # Compare the theoretical predictions with experimental results.
    plt.errorbar(Y_data, J_data, yerr = err_data, fmt = 'none', color = 'k',
                label = 'Experiment')
    plt.plot(Y_theory, J_theory, 'r-', label = 'Theory')
    plt.xlabel('$Y [\AA^{-1}]$')
    plt.ylabel('$J(Y, Q) [\AA]$')
    plt.title('Neutron Compton profile')
    plt.show()


if __name__ == "__main__":
    main()