import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset('/home/tenny/Downloads/global-dataset-of-historical-yields/maize/yield_2016.nc4')
# ndvi = ds['NDVI'].isel(time=0)

# Plotting the 'var' data
plt.figure(figsize=(10, 5))
plt.imshow(ds['var'], cmap='viridis', origin='lower', aspect='auto')

# Add labels and title
plt.colorbar(label='Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of var')
plt.savefig('output_plot.png')  # Save the plot to a file

