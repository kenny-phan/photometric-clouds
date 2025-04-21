"""
update.py

Handles time series changes to model albedo
"""

import numpy as np 

def project_to_grid(lon, lat, size, radius, center_lon=0, center_lat=0):
    """
    Project (lon, lat) to 2D grid coordinates on a size x size image,
    with the sphere centered at (center_lon, center_lat).
    """
    # Compute relative spherical coordinates
    lon_rel = ((lon - center_lon + np.pi) % (2 * np.pi)) - np.pi
    lat_rel = lat - center_lat

    # Orthographic projection
    x_proj = np.cos(lat_rel) * np.sin(lon_rel)
    y_proj = np.sin(lat_rel)

    # Only keep points on the visible hemisphere
    visible = np.cos(lat_rel) * np.cos(lon_rel) > 0

    # Convert to image coordinates
    cx = size // 2
    cy = size // 2

    # x_proj and y_proj are in range [-1, 1] → scale by radius
    x_pixel = cx + x_proj * radius
    y_pixel = cy - y_proj * radius  # minus to flip y-axis

    return x_pixel, y_pixel, visible

def update(lon, lat, lat_function, dt): #timestep dt
    rad_diff = lat_function(lat) * dt

    new_lon = lon - rad_diff
    new_lat = lat 

    return new_lon, new_lat   

def eightTermTaylorCos(x):
    cosine_eight = 1 - (0.5*(x**2)) + ((x**4)/24) #- ((x**6)/720) # + ((x**8)/40320)
    return cosine_eight

def neptune_wind(lat): #in degrees
    return -398 + 0.188*(lat**2) - 1.2e-5*(lat**4) #m/s #astropy.units

def neptune_circ_rad_s(lat): #lat in radians
    R = 24764e3 #meters
    rot_prd = 15.9663*60*60 #seconds
    return neptune_wind(np.degrees(lat)) / (R * eightTermTaylorCos(lat)) + (1 / rot_prd)

