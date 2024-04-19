import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

class LatticeBoltzmann:
    def __init__(self, Nx, Ny, rho0, tau, Nt, obstacle=None):
        self.Nx = Nx
        self.Ny = Ny
        self.rho0 = rho0
        self.tau = tau
        self.Nt = Nt
        self.obstacle = obstacle if obstacle else np.zeros((Ny, Nx))
        self.viscosity = 0.01
        self.density = np.ones((Ny, Nx)) * rho0
        self.cx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
        self.cy = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
        self.weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    def simulate(self):
        # Your simulation code here
        pass

    def visualize(self):
        plt.imshow(self.density, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.title('Fluid Density')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def add_obstacle(self, shape, position):
        if shape == 'circle':
            x, y = position
            radius = 10  # Define the radius of the circle
            # Generate a grid of points and mark those within the circle as obstacles
            X, Y = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
            self.obstacle += ((X - x)**2 + (Y - y)**2) < radius**2
        # Add support for other shapes like square, triangle, etc.

    def set_fluid_properties(self, viscosity, density):
        self.viscosity = viscosity
        self.density = np.ones((self.Ny, self.Nx)) * density

    def set_boundary_conditions(self, boundary):
        # Set boundary conditions
        pass

    def set_resolution(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny

def main():
    Nx = 400
    Ny = 100
    rho0 = 100
    tau = 0.6
    Nt = 4000

    # Create a lattice Boltzmann simulation instance
    lbm = LatticeBoltzmann(Nx, Ny, rho0, tau, Nt)

    # Add obstacles
    lbm.add_obstacle(shape='circle', position=(Nx//4, Ny//2))

    # Set fluid properties
    lbm.set_fluid_properties(viscosity=0.01, density=1.0)

    # Simulate and visualize
    lbm.simulate()
    lbm.visualize()

    # Slider for viscosity
    ax_viscosity = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider_viscosity = Slider(ax=ax_viscosity, label='Viscosity', valmin=0.0, valmax=0.1, valinit=0.01)

    # Slider for density
    ax_density = plt.axes([0.2, 0.01, 0.65, 0.03])
    slider_density = Slider(ax=ax_density, label='Density', valmin=0.0, valmax=200.0, valinit=100.0)

    def update(val):
        viscosity = slider_viscosity.val
        density = slider_density.val
        lbm.set_fluid_properties(viscosity, density)
        lbm.simulate()
        lbm.visualize()
        plt.draw()

    slider_viscosity.on_changed(update)
    slider_density.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()