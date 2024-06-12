import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PLRProcessor:

    def __init__(self):
        pass
    
    # Builds list of variables with appropriate labels
    def plr_build_var_list(self, time_var, power_var, irrad_var, temp_var, wind_var):
        final = {
            "time_var": time_var,
            "power_var": power_var,
            "irrad_var": irrad_var,
            "temp_var": temp_var,
            "wind_var": wind_var
        }
        return final

    # Removes data entries outside of irradiance and power cutoffs, 
    # fixes timestamps to specified format, and converts columns to 
    # numeric when appropriate
    def plr_cleaning(self, df, var_list, irrad_thresh, low_power_thresh, high_power_cutoff, tmst_format="%Y-%m-%d %H:%M:%S"):
        data = pd.DataFrame(df)
        data[var_list['time_var']] = pd.to_datetime(data[var_list['time_var']], format=tmst_format)

        start_date = data[var_list['time_var']].dt.date.astype(str).iloc[0]
        if data[var_list['time_var']].dt.date.astype(str).eq(start_date).all():
            data['day'] = 1
        else:
            num = 1
            prev_date = start_date
            for idx, cur_date in enumerate(data[var_list['time_var']].dt.date.astype(str)):
                if cur_date != prev_date:
                    num += 1
                data.at[idx, 'day'] = num
                prev_date = cur_date

        data['week'] = ((data['day'].astype(int) - 1) // 7.0) + 1
        data['date'] = data[var_list['time_var']].dt.date.astype(str)
        data['psem'] = ((data['day'].astype(int) - 1) // 30.0) + 1

        irrad_filter = f"{var_list['irrad_var']} >= {irrad_thresh} & {var_list['irrad_var']} <= 1500"

        if high_power_cutoff is not None:
            power_filter1 = f"{var_list['power_var']} < {high_power_cutoff}"
            dfc = data.dropna()
            dfc = dfc.query(irrad_filter)
            dfc = dfc.query(power_filter1)
        else:
            dfc = data.dropna()
            dfc = dfc.query(irrad_filter)

        power_filter2 = f"{var_list['power_var']} >= {low_power_thresh} * {dfc[var_list['power_var']].max()}"
        dfc = dfc.query(power_filter2)

        return dfc

    # Removes data entries that are greater than threshold
    def plr_saturation_removal(self, df, var_list, sat_limit=3000, power_thresh=0.99):
        data = pd.DataFrame(df)
        data = data[data[var_list['power_var']] <= float(str(sat_limit)) * power_thresh]
        return data

    # Removes rows that are outliers
    def plr_remove_outlier(self, df):
        df = pd.DataFrame(df)
        res = df[df['outlier'] == False]
        return res
    

    