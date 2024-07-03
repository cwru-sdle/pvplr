#%%
import pandas as pd
import numpy as np
import pvlib
import timezonefinder
import random
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from feature_correction import PLRProcessor
from model_comparison import PLRModel
from plr_determination import PLRDetermination
from bootstrap_uncertainty import PLRBootstrap

nrel_api_key = "ilrJbL6wg8ztrdBU9iMcZ0v9xImYPzUOnmDMJeHu"
#%%

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"]['open_rack_glass_glass']
sandia_modules = pvlib.pvsystem.retrieve_sam("SandiaMod")
cec_inverters= pvlib.pvsystem.retrieve_sam("cecinverter")
sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

sandia_module.loc["gamma_pdc"] = -.0045
sandia_module.loc["pdc0"] = 220
cec_inverter.loc["pdc0"] = 250

#%%
def timezone_determine(lat, lon): 

    tf = timezonefinder.TimezoneFinder()

    # From the lat/long, get the tz-database-style time zone name (e.g. 'America/Vancouver') or None
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    return(timezone_str)

def random_lat_lon(n=5,lat_min=36, lat_max=41, lon_min=-109, lon_max=-102):
    """
    this code produces an array with pairs lat, lon
    """
    lat = np.random.uniform(lat_min, lat_max, n).round(2)
    lon = np.random.uniform(lon_min, lon_max, n).round(2)
    lat_lon = zip(lat, lon)
    tz = [timezone_determine(lat,lon) for lat, lon in lat_lon]

    return np.array(tuple(zip(lat, lon,tz)))


#%%
np.random.seed(seed=10)
random.seed(10)

sites = random_lat_lon()

cluster_size = [20,20,20,20,20]

site_age = [10,10,10,10,10]

sites = np.column_stack((sites, cluster_size, site_age))
#%%
processor = PLRProcessor()
model = PLRModel()
det = PLRDetermination()
boot = PLRBootstrap()

for lat, lon, tz, cluster_size, site_age in sites:

    print("NEW ITERATION--------------------------------------------------------------------------------")

    #Fixing Variable Types
    lat = float(lat)
    lon = float(lon)
    cluster_size = int(cluster_size)
    site_age = int(site_age)

    #initializing df
    system_df = pd.DataFrame()

    df = pd.DataFrame()

    #metadata dictionary
    meta = {"latitude": lat,
            "longitude": lon,
            "timezone": tz,
            "gamma_pdc": -0.0045,
            "azimuth": 180,
            "tilt": 35,
            "power_dc_rated": 220.0,
            "temp_model_params":
            pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']}

    index = pd.date_range("2010-01-01", periods=87600, freq = "h", tz = tz)

    loc_pos = str(lat) + str(lon)

    try:

        #Model Chain
        location = Location(latitude=lat, longitude=lon)

        weather, metadata = pvlib.iotools.get_psm3(location.latitude, location.longitude, nrel_api_key, "rxw497@case.edu", map_variables= True)
        # temp_air - temperature
        # dni - irradiance (should i add dhi and ghi to it)
        # wind_speed
        var_list = processor.plr_build_var_list(time_var='tmst', power_var='', irrad_var='poay', temp_var='modt', wind_var='wspa')

        system = PVSystem(surface_tilt=35, surface_azimuth=180,module_parameters=sandia_module,inverter_parameters=cec_inverter,temperature_model_parameters=temperature_model_parameters, albedo=weather['albedo'])
        print(system)

    except:
        print("unable to obtain data for location")