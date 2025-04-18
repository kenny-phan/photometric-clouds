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

def neptune_wind(lat): #in degrees
    return -398 + 0.188*(lat**2) - 1.2e-5*(lat**4) #m/s #astropy.units

def soft_cos(lat, floor=0.05):
    return np.sqrt(np.cos(lat)**2 + floor**2)  # never drops below floor

def neptune_circ_rad_s(lat): #soft version
    R = 24764e3  # meters
    rot_prd = 15.9663 * 60 * 60  # seconds
    safe_cos_val = soft_cos(lat, floor=0.05)
    return 2 * np.pi * (neptune_wind(lat) / (R * safe_cos_val) + (1 / rot_prd))

def neptune_circ_rad_s(lat): #lat in radians
    R = 24764e3 #meters
    rot_prd = 15.9663*60*60 #seconds
    denom = np.clip(np.cos(lat), 0.1, 1.0)  # Avoid blow-up
    return 2 * np.pi * (neptune_wind(np.degrees(lat)) / (R * denom) + (1 / rot_prd))

