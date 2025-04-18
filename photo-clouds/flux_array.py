import numpy as np
from update import *
from numba import njit

@njit
def add_ellipse_numba(array, cx, cy, a, b, flux):
    size = array.shape[0]
    for y in range(size):
        for x in range(size):
            if ((x - cx) ** 2 / a**2 + (y - cy) ** 2 / b**2) <= 1:
                array[y, x] += flux
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
def add_surface_numba(array, size, cx, cy, r, flux):
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                array[y, x] = flux

class FluxArray:
    
    def __init__(self, size=200, alb_rad=90, baseline_flux=1, bkg_flux=0):
        self.size = size
        self.alb_rad = alb_rad
        self.baseline_flux = int(baseline_flux / (alb_rad * np.pi**2))
        self.bkg_flux = bkg_flux
        self.array = np.full((self.size, self.size), self.bkg_flux, dtype=float)

    def add_surface(self, cx=None, cy=None, r=None, flux=None):
        cx = cx if cx is not None else self.size // 2
        cy = cy if cy is not None else self.size // 2
        r = r if r is not None else self.alb_rad
        flux = flux if flux is not None else self.baseline_flux

        add_surface_numba(self.array, self.size, cx, cy, r, flux)
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
