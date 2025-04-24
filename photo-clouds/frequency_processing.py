import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import os

def make_periodogram(ls, minimum_frequency=0.01, maximum_frequency=3, samples_per_peak=3, probability=0.05, bootstrap=False):
    frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) #0.0001, 3
    if bootstrap:
        print("bootstrapping false alarm line")
        n_bootstrap = 100 #0 # also try 100000 if it's fast
        false_alarm = ls.false_alarm_level(probability, method='bootstrap', method_kwds=dict(n_bootstraps=n_bootstrap))
    else: 
        false_alarm = ls.false_alarm_level(probability, method="baluev")
    return frequency, power, false_alarm
    
def detrend_max_freq(times, flux, probability=0.05, \
                        minimum_frequency=0.01, maximum_frequency=3, samples_per_peak=3, bootstrap=False):

    ls = LombScargle(times, flux)

    frequency, power, false_alarm = make_periodogram(ls, minimum_frequency, maximum_frequency, samples_per_peak, probability, bootstrap=bootstrap)

    false_alarm_array = np.full_like(power, false_alarm)

    peaks, _ = find_peaks(power, height=false_alarm_array)
    peak_frequencies = frequency[peaks]

    long_frequency = peak_frequencies[0]

    flux_mean = np.mean(flux)
    subtraction = ls.model(times, long_frequency)
    flux_detrended = flux - subtraction + flux_mean

    ls_detrended = LombScargle(times, flux_detrended)
    frequency_dt, power_dt, false_alarm_dt = make_periodogram(ls_detrended, minimum_frequency, maximum_frequency, samples_per_peak, probability, bootstrap=bootstrap)

    best_frequency_dt = frequency_dt[np.argmax(power_dt)]

    best_period_dt = 1 / best_frequency_dt

    false_alarm_array_dt = np.full_like(power_dt, false_alarm_dt)

    peaks_dt, _ = find_peaks(power_dt, height=false_alarm_array_dt)
    peak_frequencies_dt = frequency_dt[peaks_dt]

    return frequency_dt, power_dt, false_alarm_dt, flux_detrended, peak_frequencies_dt