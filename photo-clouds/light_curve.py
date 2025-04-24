"""
light_curve.py

Generates a simulated light curve and a light curve based on peak frequencies.
"""

import numpy as np
from scipy.optimize import curve_fit

from update import *
from flux_array import *
from frequency_processing import *
from solve_wind_speed import *

def load_lines(file_name):
    lines = np.loadtxt(file_name, comments="#", delimiter=" ", unpack=False)
    x = lines[0]
    y = lines[1]
    return x, y

def make_lightcurve(times, fluxes, latitudes, longitudes=None, object_name=None):
    if object_name == "Neptune":
        lat_space, reflectance_interp = load_lines("rad_reflectance_neptune.txt")
        
    else: 
        reflectance_interp = None
        
    flux_array = FluxArray(reflectance_interp=reflectance_interp)
    
    time = len(times)
    days_elapsed = times[-1] - times[0]

    latitudes = np.radians(latitudes)
    if longitudes is None: 
        longitudes = np.full_like(latitudes, 0)
    
    lightcurve = []

    for t in range(time):
        flux_array.add_surface()

        dt = days_elapsed * 24 * 60 * 60 / time #s

        for j in range(len(latitudes)):
            lon = longitudes[j]
            lat = latitudes[j]
        
            lon, lat = update(lon, lat, neptune_circ_rad_s, dt)
        
            x, y, visible = project_to_grid(lon, lat, size=flux_array.size, radius=flux_array.alb_rad)

            if ellipse_visible_numba(lon, lat, a_rad=np.radians(6), b_rad=np.radians(2), n_points=12, center_lon=0.0, center_lat=0.0):

                visibility = np.cos(lat) * np.cos(lon)  # same logic as in project_to_grid
                adjusted_flux = fluxes[j] * visibility
                
                flux_array.add_ellipse(cx=int(x), cy=int(y), a=6, b=2, flux=adjusted_flux)

            # Save updated values
            longitudes[j] = lon
            latitudes[j] = lat

        flux_array.set_boundary()
        frame = flux_array.array.sum()

        lightcurve.append(frame)
        
    return lightcurve

def make_sine_sum(peak_frequencies_dt):
    def sine_sum(t, *params):
        result = np.zeros_like(t)
        for i in range(len(peak_frequencies_dt)):
            A = params[2*i]
            phi = params[2*i + 1]
            result += A * np.sin(2 * np.pi * peak_frequencies_dt[i] * t + phi)
        return result
    return sine_sum

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def sum_sine_curve(times_obs, flux_obs, peak_frequencies_dt):
    # Initial guesses: amplitudes = 1, phases = 0
    initial_guess = [1, 0] * len(peak_frequencies_dt)
    sine_sum = make_sine_sum(peak_frequencies_dt)
    # Fit to normalized flux
    popt, _ = curve_fit(sine_sum, times_obs, normalize(flux_obs), p0=initial_guess)
    
    # Generate the fitted curve
    return sine_sum(times_obs, *popt), popt

def run_photo_weather(times_obs, flux_obs, object_name="Neptune", detrend_count=0, fluxes=None):
    
    frequency_dt, power_dt, false_alarm_dt, flux_detrended, peak_frequencies_dt = detrend_max_freq(times_obs, flux_obs, probability=0.0001)

    for count in range(detrend_count):
        frequency_dt, power_dt, false_alarm_dt, flux_detrended, peak_frequencies_dt = detrend_max_freq(times_obs, flux_detrended, probability=0.0001)

    stack = get_wind_solutions(peak_frequencies_dt, object_name) 
    cleaned = stack[1][~np.isnan(stack[1])]

    if object_name == "Neptune":
        lats_transformed = cleaned.copy()
        lats_transformed[cleaned > 30] = -lats_transformed[cleaned > 30]
        lats_transformed[np.abs(cleaned) <= 30] = np.abs(lats_transformed[np.abs(cleaned) <= 30])

    else:
        lats_transformed = cleaned
    
    print(f"latitude solutions: {lats_transformed}")
    
    fitted_curve, popt = sum_sine_curve(times_obs, flux_detrended, stack[0])

    if fluxes is not None:
        model_lc = make_lightcurve(times_obs, fluxes=fluxes, latitudes=lats_transformed, longitudes=popt[1::2])
    else: 
        print(f"amplitudes: {popt[::2]}, longitude offsets: {popt[1::2]}")
        model_lc = make_lightcurve(times_obs, fluxes=popt[::2], latitudes=lats_transformed, longitudes=popt[1::2])
        
    return fitted_curve, model_lc