import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from update import *
from flux_array import *

def run_animation(initial_longitudes, initial_latitudes, t, a_arr=None, b_arr=None, fluxes=None):
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    
    fig, ax = plt.subplots(figsize=(5, 5))

    # Precompute cloud positions for all frames
    all_lons = []
    all_lats = []
    longitudes = initial_longitudes.copy()
    latitudes = initial_latitudes.copy()

    for i in range(len(t)):
        dt = t[1] - t[0]
        current_lons = []
        current_lats = []
        for lon, lat in zip(longitudes, latitudes):
            lon, lat = update(lon, lat, neptune_circ_rad_s, dt)
            current_lons.append(lon)
            current_lats.append(lat)
        all_lons.append(current_lons)
        all_lats.append(current_lats)
        longitudes = current_lons
        latitudes = current_lats

    def animate(i, a_arr=a_arr, b_arr=b_arr, fluxes=fluxes):
        flux = FluxArray().add_surface(flux=10)

        # Use default values if a_arr, b_arr, or fluxes are None
        if a_arr is None:
            a_arr = np.full_like(latitudes, 6)
        if b_arr is None:
            b_arr = np.full_like(latitudes, 2)
        if fluxes is None:
            fluxes = np.full_like(latitudes, 100)  # Default flux value

        # Iterate over the current longitudes and latitudes at time step i
        for j, (lon, lat) in enumerate(zip(all_lons[i], all_lats[i])):
            # Use a_arr[j], b_arr[j], and fluxes[j] for the current longitude and latitude
            a = a_arr[j]  # Semi-major axis for the j-th point
            b = b_arr[j]  # Semi-minor axis for the j-th point
            flux_value = fluxes[j]  # Flux value for the j-th point

            # Check if the ellipse is visible (adjusted for each specific lon/lat pair)
            if ellipse_visible_numba(lon, lat, a=a, b=b, n_points=20, center_lon=0.0, center_lat=0.0):
                x, y = project_to_grid(lon, lat, size=flux.size, radius=flux.alb_rad)

                visibility = np.cos(lat) * np.cos(lon)  # Basic visibility logic
                adjusted_flux = flux_value * visibility
            
                flux.add_ellipse(cx=int(x), cy=int(y), a=a, b=b, flux=flux_value).set_boundary()

        # Clear and update the plot with the new flux data
        ax.clear()
        ax.set_xlim(0, flux.size)
        ax.set_ylim(0, flux.size)
        ax.imshow(flux.array, vmin=0, vmax=20, origin='lower')
        ax.axis("off")

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(t), blit=False)
    return ani
