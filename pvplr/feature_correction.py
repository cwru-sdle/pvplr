import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression

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

    def plr_decomposition(self, data, by, freq, power_var, time_var, start_date, plot=False, plot_file=None, title=None, data_file=None):
        """
        Perform STL decomposition on time series power-predicted data.

        Args:
            data (pd.DataFrame): Input data that went through power modeling already.
            by (string): 'D' for daily, 'W' for weekly, 'M' for monthly
            freq (int): Frequency of the time series (usually 4).
            power_var (str): Name of the column containing power data.
            time_var (str): Name of the column containing time data.
            start_date (str): Start date of the time series in timestamp format.
            plot (bool, optional): Whether to create a plot. Defaults to False.
            plot_file (str, optional): File path to save the plot. Defaults to None.
            title (str, optional): Title for the plot. Defaults to None.
            data_file (str, optional): File path to save the processed data. Defaults to None.

        Returns:
            pd.DataFrame: Processed and decomposed data.
        """

        data = pd.DataFrame(data)
        total_age = range(1, int(data[time_var].max()) + 1)  
        power_sum = pd.DataFrame({'total_age': total_age})

        # Add corresponding power and sigma values to power_sum dataframe
        for j in power_sum['total_age']:
            if j in data[time_var].values:
                mask = data[time_var] == j
                power_sum.loc[j-1, 'power'] = data.loc[mask, power_var].values[0]
                power_sum.loc[j-1, 'sigma'] = data.loc[mask, 'sigma'].values[0]
            else:
                power_sum.loc[j-1, 'power'] = None
                power_sum.loc[j-1, 'sigma'] = None

        # Remove NAs at the start and end
        power_sum = power_sum.dropna(subset=['power'])

        # Define time series object of the predicted power
        ts_power = pd.Series(power_sum['power'].values, index=pd.date_range(start=start_date, periods=len(power_sum), freq=by))

        # STL decomposition
        stl_result = STL(ts_power, period=freq).fit()

        # Create DataFrame with decomposition results
        stl_data = pd.DataFrame({
            'raw': ts_power,
            'trend': stl_result.trend,
            'seasonal': stl_result.seasonal,
            'residual': stl_result.resid
        })

        stl_data['interpolated'] = stl_data['raw'].isna()
        stl_data['age'] = range(1, len(stl_data) + 1)
        stl_data['sigma'] = power_sum['sigma'].values
        stl_data['power'] = np.where(stl_data['interpolated'], np.nan, stl_data['trend'])

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(stl_data['age'], stl_data['raw'], c='blue', alpha=0.5)
            plt.scatter(stl_data['age'], stl_data['trend'], c='red')
            plt.ylim(0.9 * stl_data['raw'].min(), 1.1 * stl_data['raw'].max())
            plt.xlabel(f'Age (Pseudo {by})')
            plt.ylabel('Predicted Power (KW) Trend')
            plt.title(title)
            plt.savefig(plot_file)
            plt.close()

        reg = LinearRegression()
        reg.fit(stl_data['power'].values.reshape(-1,1), stl_data['age'].values.reshape(-1,1), sample_weight=1/stl_data['sigma'])
        m = reg.coef_[0]
        c = reg.intercept_

        # Rate of Change is slope/intercept converted to %/year
        roc = (m / c) * 12 * 100
        print(f'Calculated PLR for decomposed data: {roc.item():.5f}')

        if data_file:
            stl_data.to_csv(data_file, index=False)

        return stl_data