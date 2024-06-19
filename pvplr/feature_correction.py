import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PLRProcessor:

    def __init__(self):
        """
        Initialize the PLRProcessor instance.
        """

        pass
    
    def plr_build_var_list(self, time_var, power_var, irrad_var, temp_var, wind_var):
        """
        Builds a list of variables with appropriate labels.

        Args:
            time_var (str): The name of the time variable.
            power_var (str): The name of the power variable.
            irrad_var (str): The name of the irradiance variable.
            temp_var (str): The name of the temperature variable.
            wind_var (str): The name of the wind variable.

        Returns:
            dict: A dictionary containing the variable names with their respective labels.
        """

        final = {
            "time_var": time_var,
            "power_var": power_var,
            "irrad_var": irrad_var,
            "temp_var": temp_var,
            "wind_var": wind_var
        }
        return final
 
    def plr_cleaning(self, df, var_list, irrad_thresh, low_power_thresh, high_power_cutoff, tmst_format="%Y-%m-%d %H:%M:%S"):
        """
        Removes data entries outside of irradiance and power cutoffs, fixes 
        timestamps to specified format, and converts columns to numeric when appropriate

        Args:
            df (pd.DataFrame): The input DataFrame.
            var_list (dict): A dictionary containing the variable names.
            irrad_thresh (float): The threshold for irradiance filtering.
            low_power_thresh (float): The lower threshold for power filtering.
            high_power_cutoff (float): The upper threshold for power filtering.
            tmst_format (str, optional): The format of the timestamp. Defaults to "%Y-%m-%d %H:%M:%S".

        Returns:
            pd.DataFrame: The cleaned DataFrame
        """

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

    def plr_saturation_removal(self, df, var_list, sat_limit=3000, power_thresh=0.99):
        """
        Remove data entries that are greater than the specified saturation limit.

        Args:
            df (pd.DataFrame): The input DataFrame.
            var_list (dict): A dictionary containing the variable names.
            sat_limit (float, optional): The saturation limit. Defaults to 3000.
            power_thresh (float, optional): The power threshold. Defaults to 0.99.

        Returns:
            pd.DataFrame: The DataFrame with saturated entries removed.
        """

        data = pd.DataFrame(df)
        data = data[data[var_list['power_var']] <= float(str(sat_limit)) * power_thresh]
        return data

    def plr_remove_outlier(self, df):
        """
        Removes rows that are outliers

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with outlier rows removed.
        """
        
        df = pd.DataFrame(df)
        res = df[df['outlier'] == False]
        return res
    

    