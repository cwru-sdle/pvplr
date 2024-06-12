#from pvplr.feature_correction import PLRProcessor
#from pvplr.model_comparison import PLRModel
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn import linear_model
import matplotlib.pyplot as plt

class PLRDetermination:

    def __init__(self):
        pass

    # Normal y=mx+b equation
    def line(self, x_data, m, b):
        return (m*x_data) + b

    # Returns standard deviation of PLR value (not needed now since scipy package is used below)
    def plr_var(self, mod, X, y, per_year):
        m = mod.coef_[0]
        y_intercept = mod.intercept_
        
        # Calculate the residuals
        residuals = mod.predict(X) - y
        
        # Calculate the residual sum of squares
        rss = np.sum(residuals**2)
        
        # Calculate the degrees of freedom
        n = len(y)
        p = len(mod.coef_)
        dof = n - p
        
        # Calculate the mean squared error
        mse = rss / dof
        
        # Calculate the covariance matrix
        cov_matrix = mse * np.linalg.inv(np.dot(X.T, X))
        
        m_var = cov_matrix[0, 0]
        y_var = cov_matrix[0, 0]  # Use the same variance as m_var
        
        u = np.sqrt((per_year / y_intercept) ** 2 * m_var + ((-per_year * m) / y_intercept ** 2) ** 2 * y_var)
        
        return u * 100

    # Calculates Performance Loss Rate (PLR) using weighted linear regression with input from power predictive model
    def plr_weighted_regression(self, data, power_var, time_var, model, per_year, weight_var):
        data = pd.DataFrame(data)
        data['pvar'] = data[power_var]
        data['tvar'] = data[time_var]
        
        x = data['tvar']
        y = data['pvar'] 
            
        if weight_var is None:
            popt, pcov = curve_fit(self.line, x, y, p0 = [-2,2500], nan_policy = 'omit')
        else:
            data['wvar'] = data[weight_var]
            popt, pcov = curve_fit(self.line, x, y, p0 = [-2,2500], sigma=data['sigma'], nan_policy = 'omit')
        
        m, c = popt
        # Calculate the error using the covariance matrix
        m_err = np.sqrt(pcov[0, 0])
        c_err = np.sqrt(pcov[1, 1])
        
        # Rate of Change is slope/intercept converted to %/year
        roc = (m / c) * per_year * 100
        
        # Calculate the error in ROC using error propagation
        #print((per_year, c, m_err, m, c_err))
        #print(pcov)
        roc_err = np.sqrt( (per_year / c)**2 * m_err**2 + (-per_year * m / c**2)**2 * c_err**2) * 100
        
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

    # Calculates Performance Loss Rate (PLR) with power data separated by one year 
    def plr_yoy_regression(self, data, power_var, time_var, model, per_year, return_PLR):
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
                mod = linear_model.LinearRegression()
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

            # Return ROC or res based on return_PLR
            if return_PLR:
                return roc_df
            else:
                return res

