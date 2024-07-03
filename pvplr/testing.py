#%%
import pandas as pd
import numpy as np
import pvlib
import rdtools
import timezonefinder
import random
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

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
# define a function that scales a value based on the segment it falls in
def plr_pw_scale(scale, plr, seg_len, sample_rate =24):

    if scale < seg_len + 1:
        return  (1 - (scale * -2 * (plr) / (365* sample_rate)))
    else:
        return  ((1 - ((-3 * seg_len * plr) / (365* sample_rate))) - (scale * (plr) / (365* sample_rate)))

def plr_linear_scale(scale, plr, sample_rate =24):

    return (1 - (scale * (plr) / (365* sample_rate)))

def plr_exponential_scale(scale, plr, ts_yrs = 10, sample_rate = 24) :

    t_cons = np.log(1 - plr * ts_yrs) / (365 * sample_rate * ts_yrs)

    return (np.exp(t_cons * scale))

def plr_hyperbolic_scale(scale, plr, ts_yrs = 10, sample_rate = 24) :

    a_cons = ((1 / (1 - plr * ts_yrs)) - 1 )/ (365 * sample_rate * ts_yrs)

    return (1 / (a_cons * scale + 1))

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

plr = [.02,.025,.03,.06,.011]

cluster_size = [20,20,20,20,20]

site_age = [10,10,10,10,10]

sites = np.column_stack((sites, plr, cluster_size, site_age))
#%%
for lat, lon, tz, plr, cluster_size, site_age in sites:

    #Fixing Variable Types
    lat = float(lat)
    lon = float(lon)
    plr = float(plr)
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

    loc_pos = str(lat) +str(lon)

    try:

        #Model Chain
        location = Location(latitude=lat, longitude=lon)

        weather, metadata = pvlib.iotools.get_psm3(location.latitude, location.longitude, nrel_api_key, "rxw497@case.edu", map_variables= True)
        print('Weather---------------------------')
        print(weather)

        system = PVSystem(surface_tilt=35, surface_azimuth=180,module_parameters=sandia_module,inverter_parameters=cec_inverter,temperature_model_parameters=temperature_model_parameters, albedo=weather['albedo'])
        print(system)
        
        mc = ModelChain.with_pvwatts(system, location) 

        #Model

        model = mc.run_model(weather)
        
        simulation = mc.results.ac

        #Replicating TMY

        weather_copy = weather.copy()
        weather_list = []

        for _ in range(site_age):
            weather_list.append(weather_copy)

        weather = pd.concat(weather_list)

        weather = weather.reset_index(drop=True)

        weather.index = index 

        #Calculating POA

        solar_pos = location.get_solarposition(index)

        dni_extra = pvlib.irradiance.get_extra_radiation(solar_pos.index)

        poa =  pvlib.irradiance.get_total_irradiance(meta["tilt"], meta["azimuth"], solar_pos["apparent_zenith"], solar_pos["azimuth"], weather["dni"], weather["ghi"], weather["dhi"], albedo=weather["albedo"], dni_extra=dni_extra)

        weather["poa"] = poa["poa_global"]

        #Calculating Tcell

        weather['Tcell'] = pvlib.temperature.sapm_cell(weather.poa, weather.temp_air,
                                              weather.wind_speed, **meta['temp_model_params']) 
        
        #metadata.to_csv(path_or_buf= "../data/rwb_simulated_metadata" + str(lat) + "_" + str(lon) + ".csv")

    except:
        print("unable to obtain data for location")

    else:

        df = pd.concat([df, pd.Series(mc.results.dc.values).rename(loc_pos)], axis=1)

        cluster_df = pd.DataFrame()
        cluster_meta = pd.DataFrame(columns=['lat', 'lon', 'skew', 'plr'])
        
        for i in range(cluster_size):
            
            #Formatting the 10 yr data
            site_df = df.copy(deep=False)

            site_df_copy = site_df.copy()

            site_list = []

            for _ in range(site_age):
                site_list.append(site_df_copy)

            site_df = pd.concat(site_list)

            site_df = site_df.reset_index(drop=True)

            site_df.index = index

            site_df["simulated"] = site_df[loc_pos] 

            #Initializing PLR
            plr_noised = plr + (plr * .2 * np.random.randn())


            #Jittering lat / lon
            lat1 = lat + .0005 * lat * np.random.randn()

            lon1 = lon + .0005 * lon * np.random.randn()

            site_name = str(lat1.__round__(4)) + "_" + str(lon1.__round__(4)) + "_"  + str(plr_noised.__round__(7))

            site_df.rename(columns = {loc_pos : site_name}, inplace=True) 

            #Degradation

            site_df["scale"] = range(87600)

            skew = np.random.lognormal(0,.25)

            site_df[site_name] = site_df[site_name] * skew

            site_df[site_name] = site_df[site_name] + site_df[site_name] * .01 *  np.random.randn(365*24*site_age)

            #site_df[site_name] = site_df[site_name] * site_df["scale"].apply(plr_hyperbolic_scale, plr = plr_noised) 
            
            #Performance Ratio
            modeled_power = pvlib.pvsystem.pvwatts_dc(weather['poa'], weather['Tcell'], meta['power_dc_rated'],
                                                        meta['gamma_pdc'], 25.0 )

            # normalized, insolation = rdtools.normalize_with_expected_power(site_df[site_name],
            #                                                                 modeled_power,
            #                                                                 weather['poa'])
            
            #site_df["test"] = site_df[site_name] / site_df["simulated"]
            site_df[site_name] = site_df[site_name] / (sandia_module["pdc0"] * system.modules_per_string)

   
            #site_df['insolation'] = insolation
            #site_df["m_p"] = modeled_power       
            #site_df["norm"] = normalized
            #site_df["poa"] = weather["poa"]
            #site_df["ghi"] = weather["ghi"]

            cluster_df = pd.concat([cluster_df,site_df[site_name]], axis=1) 
            inverter_meta = pd.DataFrame([[lat1, lon1, plr, skew, plr_noised]], columns=['lat', 'lon', 'mean_plr', 'skew', 'plr'])
            cluster_meta = pd.concat([cluster_meta,inverter_meta])
        
        system_df = pd.concat([weather.iloc[:, 7:],cluster_df], axis=1)

        #system_df.to_parquet(path= "/mnt/rstor/CSE_MSE_RXF131/staging/sdle/pv-stgnn/simulated_data/lognormal_clusters/" + str(lat) + "_" + str(lon) + ".parquet")

# %%