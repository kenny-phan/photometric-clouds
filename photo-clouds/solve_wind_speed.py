from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sympy as sp

from frequency_processing import *

freq, p = sp.symbols('freq p') 
T = 3600*24
fr = 1.5032
a = -398
b = 0.188
c = -1.2e-5

R = 24764000
rot_freq = 1.5032

def eightTermTaylorCos(x):
    cosine_eight = 1 - (0.5*(x**2)) + ((x**4)/24) - ((x**6)/720) + ((x**8)/40320)
    return cosine_eight

def rad2deg(x):
    return x * 180 / 3.14159265358979323846

def wind_speed_equation(freq, a, b, c, d, R, T, fr):
    eq = ((2 * np.pi * R * eightTermTaylorCos(p) * (freq - fr))/T) - (a + b*rad2deg(p)**2 + c*rad2deg(p)**4 + d*rad2deg(p)**6)
    return eq

def polynomial(x, a, b, c, d): 
    polynomial = a + b*(x**2) + c*(x**4) + d*(x**6)
    return polynomial
    
def solveFreqLatWind(peak_frequencies, a, b, c, d, R, T, fr):
    solutions = []
    no_solution = []
    missed_ct = 0
    
    # solving the equation 
    for freq_val in peak_frequencies:
        eq = wind_speed_equation(freq_val, a, b, c, d, R, T, fr)
        success = False
        initial_guesses = np.linspace(0, np.pi/2, 20)  # List of initial guesses
        for guess in initial_guesses:
            try:
                # Solve the equation using SymPy's nsolve
                solution = sp.nsolve(eq, p, guess)  # Try different initial guesses
                p_val = float(solution) * (180 / np.pi)
                if (p_val > 90) or (p_val < -90):
                    continue
                else: 
                    solutions.append(p_val)
                    print(f"Solution at frequency {freq_val} is {p_val} degrees")
                    success = True
                    break  # Exit the loop if a solution is found
            except (sp.SympifyError) as e:
                print(f"Initial guess {guess} failed for frequency {freq_val}: {e}")
                solutions.append(np.nan)
            except Exception as e:
                print(f"{freq_val}: {e}")        
    
        if not success:
            missed_ct = missed_ct + 1
            solutions.append(np.nan)
            no_solution.append(freq_val)
            print(f"Could not find a solution for frequency {freq_val} with given guesses.")

    return solutions, no_solution, missed_ct

def stackFreqLatWind(peak_frequencies, solutions, a, b, c, d):
    freq_stack = np.array(peak_frequencies)
    lat_stack = np.array(solutions)
    wind_stack = []
    
    for solution in solutions:
        wind_speed_at_lat_sol = polynomial(solution, a, b, c, d)
        wind_stack.append(wind_speed_at_lat_sol)
    
    vstack = np.array(wind_stack)
    sol_freq = np.stack((freq_stack, lat_stack, vstack), axis=0) #frequency (1/day) = 0, latitude = 1, windspeed = 2
    return sol_freq

def getSolutions(peak_frequencies):
    voyager4_solution, voyager4_no, missed_ct1 = solveFreqLatWind(peak_frequencies, -398, 0.188, -1.2e-5, 0, 24764000, T, fr)
    voyager4_stack = stackFreqLatWind(peak_frequencies, voyager4_solution, -398, 0.188, -1.2e-5, 0)
    
    voyager6_solution, voyager6_no, missed_ct2 = solveFreqLatWind(peak_frequencies, -389, 0.153, 1.01e-5, -3.1e-9, 24764000, T, fr)
    voyager6_stack = stackFreqLatWind(peak_frequencies, voyager6_solution, -389, 0.153, 1.01e-5, -3.1e-9)
    
    h2013_solution, h2013_no, missed_ct3 = solveFreqLatWind(peak_frequencies, -325, 0.158, -1.21e-5, 0, 24764000, T, fr)
    h2013_stack = stackFreqLatWind(peak_frequencies, h2013_solution, -325, 0.158, -1.21e-5, 0)
    
    kprime2013_solution, kprime2013_no, missed_ct4 = solveFreqLatWind(peak_frequencies, -415, 2.35e-1, -2.23e-5, 0, 24764000, T, fr)
    kprime2013_stack = stackFreqLatWind(peak_frequencies, kprime2013_solution, -415, 2.35e-1, -2.23e-5, 0)
    return voyager4_stack, voyager6_stack, h2013_stack, kprime2013_stack