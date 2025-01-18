### Code to generate a 3D picture of gaussian mixture ###
### I used it to generate a picture for the report ;) ###

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aquarel import load_theme

#theme = load_theme("arctic_light")
#theme.apply()

# Define the grid
x = np.linspace(-10, 10, 200)  # Increased resolution for clarity
y = np.linspace(-10, 10, 200)
x, y = np.meshgrid(x, y)

# Define Gaussian functions with sharper peaks and clearer separation
z1 = np.exp(-((x - 2)**2 + (y - 2)**2) / 5) * 35
z2 = np.exp(-((x + 3)**2 + (y + 3)**2) / 4) * 30
z3 = np.exp(-((x - 3)**2 + (y + 5)**2) / 6) * 40

# Combined Gaussian surfaces
z = z1 + z2 + z3

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, cmap='inferno', edgecolor='none', alpha=1)
ax.axis('off')  # Turn off the axis

# Save the image
file_path = "gaussian_mixture_surface.png"
plt.savefig(file_path)
plt.close()

print(file_path)
