import numpy as np
from numba import njit

from update import *


@njit
def pixel_to_radians(pixel_value, radius):
    """
    Convert pixel value to angular distance in radians.
    """
    return np.arctan2(pixel_value, radius)


@njit
def ellipse_visible_numba(lon, lat, a, b, n_points, center_lon, center_lat, radius_px=90):
    """
    Check if any part of an ellipse along the latitudinal axis (minor axis)
    is visible in the orthographic projection centered at (center_lon, center_lat).
    The ellipse is considered visible if at least one latitudinal vertex is visible.
    """
    a_rad = pixel_to_radians(a, radius_px)  # Convert pixel to radians (minor axis)
    
    visible = False

    # Calculate the two minor axis vertices (lon, lat ± a_rad)
    minor_lat1 = lat + a_rad
    minor_lat2 = lat - a_rad

    # Function to check visibility of a point (lon, lat) in orthographic projection
    def is_visible(lon_check, lat_check):
        lon_rel = lon_check - center_lon
        if lon_rel > np.pi:
            lon_rel -= 2 * np.pi
        elif lon_rel < -np.pi:
            lon_rel += 2 * np.pi
        
        lat_rel = lat_check - center_lat

        # Calculate the angular distance
        angular_distance = np.arccos(np.sin(lat_rel) * np.sin(center_lat) + np.cos(lat_rel) * np.cos(center_lat) * np.cos(lon_rel))
        
        # If the point is within the visible hemisphere (angular distance < π/2), it is visible
        return angular_distance < np.pi / 2

    # Check visibility for the two minor axis vertices: (lon, lat ± b_rad)
    vertex1_visible = is_visible(lon, minor_lat1)
    vertex2_visible = is_visible(lon, minor_lat2)

    # The ellipse is visible if at least one vertex is visible
    if vertex1_visible or vertex2_visible:
        visible = True
    
    return visible


@njit
def add_ellipse_numba(array, cx, cy, a, b, flux):
    size = array.shape[0]
    inside_pixels = []

    # Ensure cx, cy are within bounds using modulo arithmetic for wrap-around
    cx = cx % size  # Wrap-around center x
    cy = cy % size  # Wrap-around center y

    # Single pass: collect all pixels inside the ellipse
    for y in range(size):
        for x in range(size):
            # Ellipse equation: ((x - cx) ** 2 / a**2 + (y - cy) ** 2 / b**2) <= 1
            # This checks if the point (x, y) is inside the ellipse
            if ((x - cx) ** 2 / a**2 + (y - cy) ** 2 / b**2) <= 1:
                inside_pixels.append((y, x))

    pixel_count = len(inside_pixels)

    if pixel_count == 0:
        return array  # Avoid division by zero

    flux_per_pixel = flux / pixel_count

    # Apply the flux to each pixel
    for i in range(pixel_count):
        y, x = inside_pixels[i]
        
        # Handle wrap-around for flux application (i.e., x and y can be outside the grid)
        # Apply flux to the correct grid positions
        x_mod = x % size  # Wrap-around in x direction
        y_mod = y % size  # Wrap-around in y direction

        array[y_mod, x_mod] += flux_per_pixel

    return array


@njit
def set_boundary_numba(array, cx, cy, r):
    size = array.shape[0]
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 > r ** 2:
                array[y, x] = 0
    return array

@njit
def add_surface_numba(array, size, cx, cy, r, flux, reflectance_array=None):
    u = 0.5  # Limb darkening coefficient

    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            dist2 = dx**2 + dy**2

            if dist2 <= r ** 2:
                # ---- Latitude from pixel row ----
                lat_frac = 1.0 - 2.0 * (y / (size - 1))  # +1 at top, -1 at bottom
                latitude_rad = lat_frac * (np.pi / 2)

                # ---- Reflectivity from latitude ----
                if reflectance_array is not None:
                    # Map [-π/2, π/2] → [0, len - 1]
                    lat_idx = int((latitude_rad + (np.pi / 2)) / np.pi * (len(reflectance_array) - 1))
                    reflectivity = reflectance_array[lat_idx]
                else:
                    reflectivity = 0.8 + 0.2 * np.cos(latitude_rad)

                # ---- Limb Darkening ----
                r_norm = np.sqrt(dist2) / r
                darkening = 1.0 - u * (1.0 - np.sqrt(1.0 - r_norm**2)) if r_norm < 1.0 else 0.0

                # ---- Final Flux ----
                final_flux = flux * reflectivity * darkening
                array[y, x] = final_flux

class FluxArray:
    
    def __init__(self, size=200, alb_rad=90, baseline_flux=1, bkg_flux=0, reflectance_interp=None):
        self.size = size
        self.alb_rad = alb_rad
        self.baseline_flux = int(baseline_flux / (alb_rad * np.pi**2))
        self.bkg_flux = bkg_flux
        self.array = np.full((self.size, self.size), self.bkg_flux, dtype=float)
        self.reflectance_interp = reflectance_interp

    def add_surface(self, cx=None, cy=None, r=None, flux=None):
        cx = cx if cx is not None else self.size // 2
        cy = cy if cy is not None else self.size // 2
        r = r if r is not None else self.alb_rad
        flux = flux if flux is not None else self.baseline_flux
        
        add_surface_numba(self.array, self.size, cx, cy, r, flux, self.reflectance_interp)

        # y, x = np.ogrid[:self.size, :self.size]
        # mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        # #self.array[...] = self.bkg_flux  # "reset" background
        # self.array[mask] = flux

        return self

    
    def add_ellipse(self, cx, cy, a=6, b=2, flux=0.1):
        
        self.array = add_ellipse_numba(self.array, cx, cy, a, b, flux)
        # y_grid, x_grid = np.ogrid[:self.size, :self.size]
    
        # x_diff = x_grid - cx
        # y_diff = y_grid - cy
    
        # ellipse_mask = (x_diff**2 / a**2 + y_diff**2 / b**2) <= 1
        # np.add(self.array, flux * ellipse_mask, out=self.array, where=ellipse_mask)
    
        return self

    def set_boundary(self, cx=None, cy=None, r=None):
        cx = cx if cx is not None else self.size // 2
        cy = cy if cy is not None else self.size // 2
        r = r if r is not None else self.alb_rad

        self.array = set_boundary_numba(self.array, cx, cy, r)
    
        # y_grid, x_grid = np.ogrid[:self.size, :self.size]
        # dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    
        # self.array[dist_sq > r**2] = 0
        return self
