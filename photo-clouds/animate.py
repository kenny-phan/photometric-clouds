import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from update import *
from flux_array import *

def run_animation(longitudes, latitudes, t):
    
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    
    fig, ax = plt.subplots()
    
    def animate(i):
        dt = t[1] - t[0]
        
        flux = FluxArray().add_surface(flux=10)
    
        for j in range(len(latitudes)):
            lon = longitudes[j]
            lat = latitudes[j]
    
            lon, lat = update(lon, lat, neptune_circ_rad_s, dt)

            # Wrap lon to [-π, π] (radians)
            #lon = ((lon + np.pi) % (2 * np.pi)) - np.pi
    
            x, y, visible = project_to_grid(lon, lat, size=flux.size, radius=flux.alb_rad)
            if visible:
                flux.add_ellipse(cx=int(x), cy=int(y), a=6, b=2, flux=10).set_boundary()
    
            # Save updated values
            longitudes[j] = lon
            latitudes[j] = lat
    
        ax.clear()
        ax.imshow(flux.array, vmin=0, vmax=20)
        ax.axis("off")
    
    ani = FuncAnimation(fig, animate, frames=len(t), blit=False)
    return ani