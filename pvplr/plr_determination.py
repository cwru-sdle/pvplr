""" PLR Calculation Module

This file contains a class with yoy and regression functions to calculate PLR values
after data groes through power predictive modeling. 

"""

#from pvplr.feature_correction import PLRProcessor
#from pvplr.model_comparison import PLRModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

