""" PLR Calculation Module

This file contains a class with yoy and regression functions to calculate PLR values
after data goes through power predictive modeling. 

"""

#from feature_correction import PLRProcessor
#from model_comparison import PLRModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import piecewise_regression
import matplotlib.pyplot as plt

class PLRDetermination:

    def __init__(
        self
    ):
        """
        Initialize PlRDetermination Object
        """

        pass

    def line(
        self, 
        x_data, 
        m, 
        b
    ):
        """
        Helper function that outputs a simple linear relationship for given paramaters and x-values

        Args:
            x_data (array): input data
            m (float): slope
            b (float): y-intercept
        
        Returns:
            float: output data from linear equation
        """

        return (m*x_data) + b

    def plr_var(
        self, 
        mod, 
        x, 
        y, 
        per_year
    ):
        """
        Calculate the standard deviation of the PLR value 

        Args:
            mod (LinearRegression object): The fitted linear regression model.
            X (array): The input features.
            y (array): The target values.
            per_year (float): The percentage per year.

        Returns:
            float: The standard deviation of the PLR value as a percentage.
        """

        m = mod.coef_[0]
        y_int = mod.intercept_

        # Calculate the residuals
        residuals = mod.predict(x) - y
        
        # Calculate the residual sum of squares
        rss = np.sum(residuals**2)
        
        # Calculate the degrees of freedom
        dof = len(y) - 2
        
        # Calculate the mean squared error
        mse = rss / dof

        # Calculate the covariance matrix
        X = np.hstack((np.ones((len(y), 1)), x))
        cov_matrix = np.linalg.inv(np.dot(X.T, X)) * mse

        # Extract the variances of the slope and intercept coefficients
        var_slope = cov_matrix[1, 1]
        var_intercept = cov_matrix[0, 0]

        # Calculate the standard deviation of the PLR using the delta method
        u = np.sqrt((per_year / y_int)**2 * var_slope + ((-per_year * m) / y_int**2)**2 * var_intercept)

        return u * 100

    def plr_weighted_regression(
        self, 
        data, 
        power_var, 
        time_var, 
        model, 
        per_year, 
        weight_var
    ):
        """
        Calculate the Performance Loss Rate (PLR) using weighted linear regression with input from power predictive model.

        Args:
            data (pd.DataFrame): The input data after modeling.
            power_var (str): The name of the power variable column.
            time_var (str): The name of the time variable column.
            model (str): The name of the model (Xbx, Xbx-UTC, or PVUSA).
            per_year (float): The number of time units for that by variable per year (ex. 52 for 'week')
            weight_var (str, optional): The name of the weight variable column. If None, unweighted regression is used.

        Returns:
            pd.DataFrame: A DataFrame containing the PLR, error, slope, y-intercept, model, and method.
        """

        data = pd.DataFrame(data)
        data['pvar'] = data[power_var]
        data['tvar'] = data[time_var]
        
        x = data['tvar'].values.reshape(-1,1)
        y = data['pvar'].values

        # Create a LinearRegression object
        reg = LinearRegression()

        if weight_var is None:
            reg.fit(x, y)
        else:
            data['wvar'] = data[weight_var]
            reg.fit(x, y, sample_weight=data['wvar'])

        m = reg.coef_[0]
        c = reg.intercept_

        # Rate of Change is slope/intercept converted to %/year
        roc = (m / c) * per_year * 100

        # Calculate the error using the plr_var function
        roc_err = self.plr_var(reg, x, y, per_year)
        
        # Make roc into a DataFrame
        roc_df = pd.DataFrame({'plr': [roc], 
                                'error': roc_err, 
                                'slope': m, 
                                'y-int': c,
                                'model': model})
        
        if weight_var is not None:
            roc_df['method'] = 'weighted'
        else:
            roc_df['method'] = 'unweighted'
        
        return roc_df

    def plr_yoy_regression(
        self, 
        data,
        power_var, 
        time_var, 
        model, 
        per_year, 
        return_PLR
    ):
        """
        Calculate the Performance Loss Rate (PLR) with power data separated by one year.

        Args:
            data (pd.DataFrame): The input data after modeling.
            power_var (str): The name of the power variable column.
            time_var (str): The name of the time variable column.
            model (str): The name of the model (Xbx, Xbx-UTC, or PVUSA).
            per_year (int): The number of time units per year (ex. 52 for 'week').
            return_PLR (bool): If True, returns the PLR DataFrame. If False, returns the slope data.

        Returns:
            pd.DataFrame: If return_PLR is True, returns a DataFrame containing the PLR, PLR standard deviation,
                        model, slope, y-intercept, and method. If return_PLR is False, returns the slope data.
        """

        data = pd.DataFrame(data)
        data['pvar'] = data[power_var]
        data['tvar'] = data[time_var]
        data = data.sort_values(by='tvar')

        slope_data = []

        for j in range(len(data) - per_year):
            # Select rows separated by 1 year
            p1 = data.iloc[j]
            p2 = data[data['tvar'] == p1['tvar'] + per_year]
            df = pd.concat([p1.to_frame().T, p2]).loc[:, ['pvar', 'tvar']]

            # Only measure difference if both points exist
            if not df.isnull().any().any() and len(df) == 2:
                X = df['tvar'].values.reshape(-1, 1)
                y = df['pvar'].values.reshape(-1, 1)
                mod = LinearRegression()
                reg = mod.fit(X, y)

                # Pull out the slope and intercept of the model
                m = reg.coef_[0][0]
                b = reg.intercept_[0]

                # Collect results for every point pair
                res = {'slope': m, 'yint': b, 'start': p1['tvar']}
                slope_data.append(res)

        slope_df = pd.DataFrame(slope_data)

        if slope_df.empty:
            return None
        else:
            res = slope_df.dropna()
            res['group'] = res['start'] - per_year * (res['start'] // per_year)
            res.loc[res['group'] == 0, 'group'] = per_year
            res['year'] = res['start'] // per_year + 1

            ss = res['slope'].median()
            yy = res['yint'].median()
            roc = (ss / yy) * 100 * per_year

            roc_df = pd.DataFrame({'plr': [roc],
                            'plr_sd': np.array([(res['slope'] / yy) * 100 * per_year]).std(),
                            'model': model,
                            'slope': m, 
                            'y-int': b,
                            'method': 'year-on-year'})

            # Return ROC or res based on return_PLR input
            if return_PLR:
                return roc_df
            else:
                return res
    
    def plr_piecewise(
        self,
        df, 
        power_model,
        n_breakpoints,
        per_year, 
        power_var, 
        time_var, 
        return_model = False,
        plot = False
    ):
        """
        Perform piecewise linear regression on the given data.

        Parameters:
        df (pd.DataFrame): Data frame of corrected power measurements
        power_model (str): Name of Power Predictive Model used
        n_breakpoints (int): Number of desired breakpoints
        per_year (int): 365 for daily, 52 for weekly, 12 for monthly
        power_var (str): Name of the power variable column
        time_var (str): Name of the time variable column
        return_model (bool): If True, return model summary stats; if False, return PLR results
        plot (bool): If True, returns plot of piecewise linear power model

        Returns:
        dict: Results of the piecewise linear regression

        Uses piecewise-regression package: https://joss.theoj.org/papers/10.21105/joss.03859
        """

        # Extract x and y
        x = df[time_var].values
        y = df[power_var].values

        pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=n_breakpoints)

        # Total Modeling Summary Statistics
        if return_model:
            return pw_fit.summary()

        if per_year == 12:
            by = 'month'
        if per_year == 52:
            by = 'week'
        if per_year == 365:
            by = 'day'

        # Create Pandas Dataframe for results
        segments_data = pd.DataFrame(columns=['segment', 'seg_start', 'seg_end', 'slope', 'y-int', 'plr', 'plr_sd'])

        results = pw_fit.get_results()
        data = pw_fit.get_params()

        # segment start and ending points
        breakpoints = [v for k, v in data.items() if k.startswith('breakpoint')]
        breakpoints.insert(0, 0)
        breakpoints.append(x.max())

        # slope
        alphas = [v for k, v in data.items() if k.startswith('alpha')]
        slope_err = []
        for key, value in results['estimates'].items():
            if key.startswith('alpha'):
                slope_err.append(value['se'])

        # y-int
        y_int = results['estimates']['const']['estimate']
        y_int_err = results['estimates']['const']['se']
        
        for i in range(len(breakpoints)-1):
            segments_data.loc[i, 'segment'] = i + 1
            segments_data.loc[i, 'seg_start'] = breakpoints[i]
            segments_data.loc[i, 'seg_end'] = breakpoints[i + 1]
            segments_data.loc[i, 'slope'] = alphas[i]
            segments_data.loc[i, 'y-int'] = y_int
            segments_data.loc[i, 'plr'] = ((alphas[i])/y_int) * 100 * per_year 
            segments_data.loc[i, 'plr_sd'] = np.sqrt((per_year / y_int)**2 * slope_err[i] + ((-per_year * alphas[i]) / y_int**2)**2 * y_int_err)
        
        if plot:
            self.plot_piecewise(time=x, power=y, pw_fit=pw_fit, power_model=power_model, by=by)

        return segments_data
    
    def plot_model(
        self, 
        df, 
        power_model, 
        by
    ):
        """
        Make a scatter plot of the power predictive model results along with PLR best fit line.

        Args:
            df (pd.DataFrame): DataFrame containing the model data.
            power_model (str): Name of the power model being plotted.
            by (str): Time unit for x-axis ('day', 'week', 'month').
        """

        df = pd.DataFrame(df)
        time = df.index.to_list()
        power = df['power_var']

        if by == 'day':
            per_year = 365
        if by == 'week':
            per_year = 52
        if by == 'month':
            per_year = 12

        # PLR Calculation with Weighted Regression
        plr_df = self.plr_weighted_regression(data=df, power_var='power_var', time_var='time_var', model=power_model, per_year=per_year, weight_var='sigma')
        print(plr_df)
        slope = plr_df['slope'].item()
        y_int = plr_df['y-int'].item()

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor('#eee6ff')
        ax.scatter(time, power, color='#0000b3', label='Predicted Power data')
        ax.plot(time, [t*slope + y_int for t in range(len(time))], color='black', label='PLR best-fit line')
        ax.set_ylim(0.8*min(power), 1.2*max(power))
        ax.set_xticks(np.linspace(0,max(time),10)//1)
        
        ax.grid(True, linestyle='-', linewidth=2, alpha=0.5, color='white')
        ax.set_xlabel(f"Time (in {by}s)")
        ax.set_ylabel('Power (in kilowatts)')
        ax.legend()
        plt.title(f"{power_model} Power Predictive Model Results")
        plt.tight_layout()
        plt.show()

    def plot_piecewise(
        self,
        time,
        power,
        pw_fit,
        power_model,
        by,
    ):
        """
        Make a scatter plot of the power predictive model results along with piecewise PLR best fit lines.

        Args:
            df (pd.DataFrame): DataFrame containing the model data.
            time (array): Array of time values
            power (array): Array of power values
            pw_fit (piecewise_regression object): piecewise_regression results object
            power_model: (str): Name of the power model being plotted.
            by (str): Time unit for x-axis ('day', 'week', 'month').
        """

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor('#eee6ff')
        
        plt.sca(ax)
        pw_fit.plot_data(color='#0000b3', s=20, label="Decomposed Power Data")
        pw_fit.plot_fit(color="black", linewidth=1, label="PLR best-fit line")
        pw_fit.plot_breakpoints()
        pw_fit.plot_breakpoint_confidence_intervals()
        
        ax.grid(True, linestyle='-', linewidth=2, alpha=0.5, color='white')
        ax.set_xlabel(f"Time (in {by}s)")
        ax.set_ylabel("Power (in kW)")
        ax.set_ylim(0.8*min(power), 1.2*max(power))
        ax.set_xticks(np.linspace(0, max(time), 10)//1)
        ax.legend()

        plt.title(f"{power_model} Power Predictive Model Results with Piecewise PLR")
        plt.tight_layout()
        plt.show()


