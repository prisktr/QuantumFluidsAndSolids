Contents:
  -- requirements.txt: This file lists all needed python libraries.
  -- He4_calc.py: This is the main module.  The script calculates a theoretical
  prediction for the neutron Compton profile and then compares it to
  experimental data.
  -- He4_function_test_script.py: This script is intended to check that the
  various functions yield physically reasonable and correct results.  A number
  of curves are generated and then plotted.
  -- He4_data.py: This module contains numerical data required for the program.
  -- He4_functions.py: This module is a function library for the program.
  -- x_27.0.npy, rGammas_27.0.npy, iGammas_27.0.npy: Numerical data needed to
  run the test script.

Notes:
  -- To install needed packages, use: pip install -r requirements.txt
  -- The main module requires ~1 core-hour to finish.
