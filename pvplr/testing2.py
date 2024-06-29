import pvlib
import pandas as pd

# Set the location (latitude, longitude, timezone)
latitude, longitude = 40.7128, -74.0060  # New York City coordinates
tz = 'America/New_York'

# Create a location object
location = pvlib.location.Location(latitude, longitude, tz=tz)

# Set the time period for the data
start = pd.Timestamp('2021-01-01', tz=tz)
end = pd.Timestamp('2021-12-31', tz=tz)

# Get TMY data for the location
tmy_data, tmy_info, *_ = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)

# Extract relevant columns
temperature = tmy_data['temp_air']
wind_speed = tmy_data['wind_speed']
ghi = tmy_data['ghi']

# Create a DataFrame with the extracted data
df = pd.DataFrame({
    'Temperature (°C)': temperature,
    'Wind Speed (m/s)': wind_speed,
    'Global Horizontal Irradiance (W/m²)': ghi
})

# Display the first few rows of the data
print(df)
