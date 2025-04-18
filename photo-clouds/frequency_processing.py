import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import os

def make_periodogram(ls, minimum_frequency, maximum_frequency=3, samples_per_peak=10, probability=0.05, bootstrap=False):
    frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) #0.0001, 3
    if bootstrap:
        print("bootstrapping false alarm line")
        n_bootstrap = 1000 #0 # also try 100000 if it's fast
        false_alarm = ls.false_alarm_level(probability, method='bootstrap', method_kwds=dict(n_bootstraps=n_bootstrap))
    else: 
        false_alarm = ls.false_alarm_level(probability)
    return frequency, power, false_alarm
    
def detrend_max_freq(times, flux, probability=0.05, \
                        minimum_frequency=0.01, maximum_frequency=3, samples_per_peak=10):
    #print("Now processing periodogram...")

    ls = LombScargle(times, flux)

    frequency, power, false_alarm = make_periodogram(ls, minimum_frequency, maximum_frequency, samples_per_peak, probability)
    print(false_alarm)
    false_alarm_array = np.full_like(power, false_alarm)

    peaks, _ = find_peaks(power, height=false_alarm_array)
    peak_frequencies = frequency[peaks]
    #print("Peak frequencies:", peak_frequencies)

    peak_periods = 1 / peak_frequencies

    #print("Peak periods:", peak_periods)

    long_period = peak_periods[0]
    long_frequency = peak_frequencies[0]

    flux_mean = np.mean(flux)
    subtraction = ls.model(times, long_frequency)
    flux_detrended = flux - subtraction + flux_mean

    ls_detrended = LombScargle(times, flux_detrended)
    frequency_dt, power_dt, false_alarm_dt = make_periodogram(ls_detrended, minimum_frequency, maximum_frequency, samples_per_peak, probability)

    best_frequency_dt = frequency_dt[np.argmax(power_dt)]

    best_period_dt = 1 / best_frequency_dt

    false_alarm_array_dt = np.full_like(power_dt, false_alarm_dt)

    peaks_dt, _ = find_peaks(power_dt, height=false_alarm_array_dt)
    peak_frequencies_dt = frequency_dt[peaks_dt]
    #print("Peak detrended frequencies:", peak_frequencies_dt)

    peak_periods_dt = 1 / peak_frequencies_dt

    #print("Peak detrended periods:", peak_periods_dt)

    return frequency_dt, power_dt, false_alarm_dt, flux_detrended, peak_frequencies_dt