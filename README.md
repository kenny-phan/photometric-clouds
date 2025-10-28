# photo-clouds

[10/28/25 UPDATE] This code was cool, but is built on the fauly assumption that all storms in a gas giant atmosphere are stable over a span of months. This is not the case! Instead, features form and dissipate at shorter timescales and at different latitudes (thus leading to multiple perodicities in a light curve periodogram), rather than all periodicity-causing features being present at the same time.

A Python codebase that inputs a planetary wind speed profile and optionally a time and flux light curve array and outputs a lightcurve of the base frequencies and a simulated light curve from a basic model of clouds passing over the albedo. 

Time and flux arrays may be input as a .txt file with np.loadtxt(file_name, comments="#", delimiter=" ", unpack=False).

