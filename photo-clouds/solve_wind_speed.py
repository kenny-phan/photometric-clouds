"""
solve_wind_speed.py

Solves for latitude locations and wind speeds at those locations assuming peak frequencies are due to a moving cloud against the albedo.
"""

import sys

import numpy as np
import sympy as sp

from frequency_processing import *

freq, p = sp.symbols('freq p') 

# homemade functions to utilize sympy solvers
def eightTermTaylorCos(x):
    cosine_eight = 1 - (0.5*(x**2)) + ((x**4)/24) - ((x**6)/720) + ((x**8)/40320)
    return cosine_eight


def rad2deg(x):
    return x * 180 / 3.14159265358979323846


def polynomial(x, a, b, c): 
    return a + b*(x**2) + c*(x**4)


# wind speed frequency model minus wind speed equation to solve for latitude p
def neptune_wind_speed(freq, a, b, c, R, T, fr):
    eq = ((2 * np.pi * R * eightTermTaylorCos(p) * (freq - fr))/T) - polynomial(rad2deg(p), a, b, c)
    return eq

from scipy.optimize import root_scalar

def solve_numeric_latitude(freq_val, a, b, c, R, T, fr):
    def equation(p_rad):
        cos8 = 1 - 0.5*p_rad**2 + (p_rad**4)/24 - (p_rad**6)/720 + (p_rad**8)/40320
        deg = np.degrees(p_rad)
        lhs = (2 * np.pi * R * cos8 * (freq_val - fr)) / T
        rhs = a + b * deg**2 + c * deg**4
        return lhs - rhs

    # Try solving in the range 0 to pi/2 radians
    result = root_scalar(equation, bracket=[0, np.pi / 2], method='brentq')
    if result.converged:
        return np.degrees(result.root)
    return np.nan


def solve_freq_lat_wind(peak_frequencies, a, b, c, R, T, fr):
    """
    Attempts to solve for planetary latitude (in degrees) given a set of peak frequencies,
    using a wind speed model and symbolic equation solving.

    Args:
        peak_frequencies (list or np.ndarray): List of frequency values to evaluate.
        a, b, c (float): Coefficients in the wind speed model.
        R (float): Planetary radius.
        T (float): Orbital period.
        fr (float): Frame rotation parameter.

    Returns:
        tuple: (solutions, no_solution, missed_ct)
            - solutions (list): List of latitude solutions in degrees (or NaN if unsolved).
            - no_solution (list): Frequencies for which no solution was found.
            - missed_ct (int): Number of frequencies with no valid solution.
    """
    solutions = []
    no_solution = []
    missed_ct = 0

    for freq_val in peak_frequencies:
        try:
            lat = solve_numeric_latitude(freq_val, a, b, c, R, T, fr)
            if -90 <= lat <= 90:
                solutions.append(lat)
            else:
                solutions.append(np.nan)
                missed_ct += 1
                no_solution.append(freq_val)
        except:
            solutions.append(np.nan)
            missed_ct += 1
            no_solution.append(freq_val)

    return solutions, no_solution, missed_ct
    

def stack_freq_lat_wind(peak_frequencies, solutions, a, b, c):
    lat_stack = np.array(solutions)
    freq_stack = np.array(peak_frequencies)
    wind_stack = polynomial(lat_stack, a, b, c)
    return np.stack((freq_stack, lat_stack, wind_stack), axis=0)


def get_wind_solutions(peak_frequencies, object_name):
    try:
        if object_name == "Neptune":
            T = 3600*24
            fr = 1.5032
            a = -398
            b = 0.188
            c = -1.2e-5
            R = 24764000

        else: 
            raise ValueError("meowww! use either 'Neptune' or 'Uranus'")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    solution, no, missed_ct = solve_freq_lat_wind(peak_frequencies, a, b, c, R, T, fr)
    stack = stack_freq_lat_wind(peak_frequencies, solution, a, b, c)
    
    return stack