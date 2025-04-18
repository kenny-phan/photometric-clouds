import numpy as np
from scipy.optimize import curve_fit
from update import *
from flux_array import *
from frequency_processing import *
from solve_wind_speed import *

def load_lines(file_name):
    lines = np.loadtxt(file_name, comments="#", delimiter=" ", unpack=False)
    times = lines[0]
    flux = lines[1]
    return times, flux

def make_lightcurve(flux_array, latitudes, fluxes, time=2575, days_elapsed=24.299219225067645):
    longitudes = np.full_like(latitudes, 0)
    latitudes = np.radians(latitudes)
    lightcurve = []

    for t in range(time):
        flux_array.add_surface()

        dt = days_elapsed * 24 * 60 * 60 / time #s
        # visible_count = 0
        for j in range(len(latitudes)):
            lon = longitudes[j]
            lat = latitudes[j]
        
            lon, lat = update(lon, lat, neptune_circ_rad_s, dt)
            #print(f"Updated lon: {lon}, lat: {lat}")
        
            x, y, visible = project_to_grid(lon, lat, size=flux_array.size, radius=flux_array.alb_rad)
            #print(f"Visible: {visible}")

            if visible:
                flux_array.add_ellipse(cx=int(x), cy=int(y), a=6, b=2, flux=fluxes[j])
                #print(f"⚠️ Out-of-bounds projection at t={t}: x={x:.1f}, y={y:.1f}")
                
                # print(f"Adding flux at ({int(x)}, {int(y)}) with flux = {fluxes[j]}")
                # visible_count += 1

            # Save updated values
            longitudes[j] = lon
            latitudes[j] = lat

        # if t % 10 == 0:
        #     plt.imshow(flux_array.array)
        #     plt.title(f"Frame {t}")
        #     plt.show()
                
        # print(f"🛰️ {visible_count}/{len(latitudes)} features visible at t={t}")
        flux_array.set_boundary()
        frame = flux_array.array.sum()
        # print(f"Frame sum at t={t}: {flux_array.array.sum()}")

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

def run_photo_weather(times_obs, flux_obs):
    frequency_dt, power_dt, false_alarm_dt, flux_detrended, peak_frequencies_dt = detrend_max_freq(times_obs, flux_obs, probability=0.0001)
    stack, a, b, c = getSolutions(peak_frequencies_dt)
    cleaned = stack[1][~np.isnan(stack[1])]
    fitted_curve, popt = sum_sine_curve(times_obs, flux_obs, stack[0])
    flux_array = FluxArray().add_surface()
    lc = make_lightcurve(flux_array, latitudes=cleaned, fluxes=popt[::2], time=2575, days_elapsed=24.299219225067645)
    return fitted_curve, lc