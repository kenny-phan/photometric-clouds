import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from update import *
from flux_array import *

def run_animation(initial_longitudes, initial_latitudes, t):
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    
    fig, ax = plt.subplots()

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

    def animate(i):
        flux = FluxArray().add_surface(flux=10)
        for lon, lat in zip(all_lons[i], all_lats[i]):
            if ellipse_visible_numba(lon, lat, a_rad=np.radians(6), b_rad=np.radians(2), n_points=12, center_lon=0.0, center_lat=0.0):
                x, y, _ = project_to_grid(lon, lat, size=flux.size, radius=flux.alb_rad)
                flux.add_ellipse(cx=int(x), cy=int(y), a=6, b=2, flux=100).set_boundary()
        ax.clear()
        ax.imshow(flux.array, vmin=0, vmax=20, origin='lower')
        ax.axis("off")
        print(f"Frame {i}: {np.sum([ellipse_visible_numba(lon, lat, 6/flux.alb_rad, 2/flux.alb_rad, 12, 0.0, 0.0) for lon, lat in zip(all_lons[i], all_lats[i])])} visible clouds")

    ani = FuncAnimation(fig, animate, frames=len(t), blit=False)
    return ani
