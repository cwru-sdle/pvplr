#from pvplr.feature_correction import PLRProcessor
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

class PLRModel: 
    
    def __init__(self):
        pass

    # Helper function to initialize model dataframe
    def model_initialization(self, df, var_list, by):
        data = pd.DataFrame(df)

        if var_list["wind_var"] is None:
            del var_list["wind_var"]
        else: 
            wind_var = var_list["wind_var"]
        
        model_df = data[list(var_list.values())].copy()
        model_df.columns = list(var_list.keys())

        if by == "month":
            model_df['tvar'] = data['psem']
        elif by == "week":
            model_df['tvar'] = data['week']
        elif by == "day":
            model_df['tvar'] = data['day']
        else:
            raise ValueError("Error: 'By' set to something other than 'month', 'week', or 'day'. Try again with one of these values")

        return model_df

    # Helper function to generate predicted data or assign dataframe to pred input
    def generate_predicted_data(self, model_df, var_list, predict_data):
        if predict_data is None:
            n = model_df.groupby("tvar").size().reset_index(name="n")
            pred = []
            split_var = 'tvar'

            # Iterate over each group in the DataFrame
            groups = model_df.groupby(split_var).groups
            grouped_dfs = [model_df.loc[group] for group in groups.values()]

            # Iterating variables
            start = 0
            end = n["n"][0]

            # Iterate over each DataFrame in the list
            for i, df in enumerate(grouped_dfs, start=0):

                mod = model_df[start:end]
                start = end
                end = end + n["n"][i]

                # miniumum value of maximums that are greater than 300 in irrad_var column
                irrad_var = (
                    mod
                    .groupby('tvar')
                    .agg({'irrad_var': 'max'})
                    .reset_index()
                    .query('irrad_var > 300')
                    .agg({'irrad_var': 'min'})
                    .iloc[0]
                )

                #take average of all temps
                temp_var = mod['temp_var'].mean()

                #take average of all wind speeds
                if var_list['wind_var'] is not None:
                    wind_var = mod['wind_var'].mean()
                else:
                    wind_var = None

                # build pred
                if 'wind_var' in model_df.columns:
                    pred.append([irrad_var, temp_var, wind_var])
                else:
                    pred.append([irrad_var, temp_var])
            
            if 'wind_var' in model_df.columns:
                pred = pd.DataFrame(pred, columns=['irrad_var', 'temp_var', 'wind_var'])
            else:
                pred = pd.DataFrame(pred, columns=['irrad_var', 'temp_var'])
        
        else:
            pred = predict_data

        pred = pred.dropna()

        return pred   

    # Xbx - groups data by the specified time interval and performs a linear regression 
    # using the formula: P_{pred = beta_0 + beta_1*G + beta_2*T + epsilon.
    def plr_xbx_model(self, df, var_list, by, data_cutoff, predict_data): 
        # initialize dataframes from helper function above
        model_df = self.model_initialization(df=df, var_list=var_list, by=by)

        # get predicted data from helper function above
        pred = self.generate_predicted_data(model_df=model_df, var_list=var_list, predict_data=predict_data)

        # determine number of data points for each time period
        n = model_df.groupby("tvar").size().reset_index(name="n")

        res_dfs = []
        split_var = 'tvar'

        # Iterate over each group in the DataFrame
        groups = model_df.groupby(split_var).groups
        grouped_dfs = [model_df.loc[group] for group in groups.values()]

        # Iterating variables
        start = 0
        end = n["n"][0]

        # Iterate over each DataFrame in the list
        for i, df in enumerate(grouped_dfs, start=0):
            
            mod = model_df[start:end]
            start = end
            end = end + n["n"][i]

            mod = mod.dropna()
            if 'wind_var' in model_df.columns:
                X = mod[['irrad_var', 'temp_var', 'wind_var']]
            else:
                X = mod[['irrad_var', 'temp_var']]
            y = mod['power_var']

            # Create a linear regression model
            model = linear_model.LinearRegression()
            # Fit the model to the data

            if not X.empty:
                model.fit(X, y)

                # Create a new DataFrame with the results
                res = pd.DataFrame({
                    'tvar': df['tvar'].iloc[0],
                    'intercept': model.intercept_,
                    'irrad_coef': model.coef_[0],
                    'temp_coef': model.coef_[1],
                    'wind_coef': [model.coef_[2]] if 'wind_var' in model_df.columns else [None],
                    'prediction': [model.predict(pred[['irrad_var', 'temp_var', 'wind_var']] if 'wind_var' in model_df.columns else pred[['irrad_var', 'temp_var']])[0]],
                    'r_squared': [model.score(X, y)],
                    'adj_r_squared': [1 - (1 - model.score(X, y)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)],
                    'mse': [mean_squared_error(y, model.predict(X))], 
                    # 'sigma': [np.sqrt(mean_squared_error(y, model.predict(X)))]
                }, index=[0])
                
                res_dfs.append(res)

        res_dfs = pd.concat(res_dfs, ignore_index=True)
        res_dfs = pd.merge(res_dfs, n, on='tvar', how='left')
        res_dfs = res_dfs[res_dfs['n'] >= data_cutoff]
        res_dfs['tvar'] = pd.to_numeric(res_dfs['tvar'])
        res_dfs['std_error'] = np.sqrt(res_dfs['mse']) / np.sqrt(n['n'])

        fitted_values = res_dfs['prediction']
        iqr = stats.iqr(fitted_values)
        lower = np.quantile(fitted_values, 0.25) - 1.5 * iqr
        upper = np.quantile(fitted_values, 0.75) + 1.5 * iqr
        res_dfs['outlier'] = (res_dfs['prediction'] > upper) | (res_dfs['prediction'] < lower) 

        res_dfs = res_dfs.assign(
            time_var=res_dfs['tvar'],
            power_var=res_dfs['prediction'],
            sigma=res_dfs['std_error'] * np.sqrt(res_dfs['n'])
        ) [['time_var', 'power_var', 'std_error', 'sigma', 'outlier']]

        return res_dfs
    
    # Xbx-UTC - groups data by the specified time interval and performs a linear regression 
    # using the formula: power_corr ~ irrad_var - 1. Predicted values of irradiance, 
    # temperature, and wind speed (if applicable) are added for reference. The function uses 
    # a universal temperature correction, rather than the monthly regression correction 
    def plr_xbx_utc_model(self, df, var_list, data_cutoff, predict_data, by="month", ref_irrad=900, irrad_range=10):
        # initialize dataframes from helper function above
        model_df = self.model_initialization(df=df, var_list=var_list, by=by)

        # get predicted data from helper function above
        pred = self.generate_predicted_data(model_df=model_df, var_list=var_list, predict_data=predict_data)

        # Universal Temperature Coefficient Calculation
        utc = model_df[(model_df['irrad_var'] < (ref_irrad + irrad_range)) & 
                        (model_df['irrad_var'] > (ref_irrad - irrad_range)) & 
                        (model_df['power_var'] > 0.05 * model_df['power_var'].max())].copy()
        utc.loc[:, 'frac'] = utc['power_var'] * (utc['temp_var'] + 273.15)  # kelvin

        # filter outliers that may influence coefficient
        Q1 = utc['frac'].quantile(0.25)
        Q3 = utc['frac'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)
        utc = utc[(utc['frac'] >= lower) & (utc['frac'] <= upper)]
        
        model = linear_model.LinearRegression()
        model.fit(utc[['temp_var']], utc['power_var'])
        utc_coeff = model.coef_[0] / ref_irrad

        res_dfs = []
        split_var = 'tvar'
        n = model_df.groupby("tvar").size().reset_index(name="n")

        groups = model_df.groupby(split_var).groups
        grouped_dfs = [model_df.loc[group] for group in groups.values()]

        # Iterating variables
        start = 0
        end = n["n"][0]

        for i, df in enumerate(grouped_dfs, start=0):

            if i >= pred.shape[0]:
                break

            if i not in pred.index:
                print(f"Warning: Index {i} does not exist in the pred DataFrame. Skipping this iteration.")
                continue

            mod = model_df[start:end]
            start = end
            end = end + n["n"][i]

            if not mod.empty :  

                power_corr = mod['power_var'].values + utc_coeff * (np.full(mod['temp_var'].shape, pred['temp_var'][i]) - mod['temp_var'].values) * mod['irrad_var'].values

                X = mod['irrad_var'].values.reshape(-1, 1)
                y = power_corr.reshape(-1, 1)
                
                model = linear_model.LinearRegression(fit_intercept=False)
                model.fit(X, y)
                prediction = model.predict(pred['irrad_var'].values.reshape(-1, 1))
                #tidy = pd.Series(model.coef_[0], index=['irrad_var'])
                
                res = pd.DataFrame({
                    'tvar': mod['tvar'].iloc[0],
                    'prediction': prediction[0],
                    'temp_coef': model.coef_[0],
                    'r_squared': model.score(X, y),
                    'adj_r_squared': 1 - (1 - model.score(X, y)) * (len(y) - 1) / (len(y) - X.shape[1] - 1),
                    'mse': [mean_squared_error(y, model.predict(X))], 
                    # 'sigma': np.sqrt(mean_squared_error(y, model.predict(X)))
                }, index=[0])
                            
                res_dfs.append(res)
            
        res_dfs = pd.concat(res_dfs, ignore_index=True)
        res_dfs = pd.merge(res_dfs, n, on='tvar', how='left')
        res_dfs = res_dfs[res_dfs['n'] >= data_cutoff]
        res_dfs['tvar'] = pd.to_numeric(res_dfs['tvar'])
        res_dfs['std_error'] = np.sqrt(res_dfs['mse']) / np.sqrt(n['n'])

        fitted_values = res_dfs['prediction']
        iqr = stats.iqr(fitted_values)
        lower = np.quantile(fitted_values, 0.25) - 1.5 * iqr
        upper = np.quantile(fitted_values, 0.75) + 1.5 * iqr
        res_dfs['outlier'] = res_dfs['prediction'].apply(lambda x: (x > upper) | (x < lower))

        res_dfs = res_dfs.assign(
            time_var=res_dfs['tvar'],
            power_var=res_dfs['prediction'],
            sigma=res_dfs['std_error'] * np.sqrt(res_dfs['n']),
            outlier=res_dfs['outlier'],
        )[['time_var', 'power_var', 'std_error', 'sigma', 'outlier']]

        return res_dfs

    # PVUSA - a physics-based model that groups data and performs a linear regression according to
    # the formula P = G_POA * (beta_{0} + beta_{1}*G + beta_{2}*T_{amb} + beta_{3}*W
    def plr_pvusa_model(self, df, var_list, by, data_cutoff, predict_data):
        # initialize dataframes from helper function above
        model_df = self.model_initialization(df=df, var_list=var_list, by=by)

        # get predicted data from helper function above
        pred = self.generate_predicted_data(model_df=model_df, var_list=var_list, predict_data=predict_data)

        # determine number of data points for each time period
        n = model_df.groupby("tvar").size().reset_index(name="n")

        res_dfs = []
        split_var = 'tvar'

        # Iterate over each group in the DataFrame
        groups = model_df.groupby(split_var).groups
        grouped_dfs = [model_df.loc[group] for group in groups.values()]

        # Iterating variables
        start = 0
        end = n["n"][0]

        # Iterate over each DataFrame in the list
        for i, df in enumerate(grouped_dfs, start=0):

            if i>=pred.shape[0]:
                break

            if i not in pred.index:
                print(f"Warning: Index {i} does not exist in the pred DataFrame. Skipping this iteration.")
                continue

            mod = model_df[start:end]
            start = end
            end = end + n["n"][i]
            
            if 'wind_var' in model_df.columns:
                irrad_corr = mod['irrad_var'].values + (mod['irrad_var'].values)**2 + (mod['irrad_var'].values * mod['temp_var'].values) + (mod['irrad_var'].values * mod['wind_var'].values)
                pred_irrad_corr = pred['irrad_var'][i] + (pred['irrad_var'][i])**2 + (pred['irrad_var'][i] * pred['temp_var'][i]) + (pred['irrad_var'][i] * pred['wind_var'][i])
            else:
                irrad_corr = mod['irrad_var'].values + (mod['irrad_var'].values)**2 + (mod['irrad_var'].values * mod['temp_var'].values)
                pred_irrad_corr = pred['irrad_var'][i] + (pred['irrad_var'][i]**2 + (pred['irrad_var'][i] * pred['temp_var'][i]))
            
            X = irrad_corr.reshape(-1, 1)
            y = mod['power_var'].values.reshape(-1, 1)

            # Create a linear regression model
            model = linear_model.LinearRegression()
            # Fit the model to the data
            model.fit(X, y)

            # Create a new DataFrame with the results
            res = pd.DataFrame({
                'tvar': df['tvar'].iloc[0],
                'intercept': model.intercept_[0],  
                'coeff': model.coef_[0][0],  
                'prediction': model.predict([[pred_irrad_corr]])[0], 
                'r_squared': model.score(X, y),
                'adj_r_squared': 1 - (1 - model.score(X, y)) * (len(y) - 1) / (len(y) - X.shape[1] - 1),
                # 'sigma': np.sqrt(np.mean((y - model.predict(X))**2)),  
                'mse': mean_squared_error(y, model.predict(X))
            }, index=[0])
            
            res_dfs.append(res)
        
        res_dfs = pd.concat(res_dfs, ignore_index=True)
        res_dfs = pd.merge(res_dfs, n, on='tvar', how='left')
        res_dfs = res_dfs[res_dfs['n'] >= data_cutoff]
        res_dfs['tvar'] = pd.to_numeric(res_dfs['tvar'])
        res_dfs['std_error'] = np.sqrt(res_dfs['mse']) / np.sqrt(n['n'])

        fitted_values = res_dfs['prediction']
        iqr = stats.iqr(fitted_values)
        lower = np.quantile(fitted_values, 0.25) - 1.5 * iqr
        upper = np.quantile(fitted_values, 0.75) + 1.5 * iqr
        res_dfs['outlier'] = res_dfs['prediction'].apply(lambda x: (x > upper) | (x < lower))

        res_dfs = res_dfs.assign(
            time_var=res_dfs['tvar'],
            power_var=res_dfs['prediction'],
            sigma=res_dfs['std_error'] * np.sqrt(res_dfs['n'])
        )[['time_var', 'power_var', 'std_error', 'sigma', 'outlier']]

        return res_dfs
    
    # 6k - groups data by time interval and performs a linear regression according to the
    # formula: power_var ~ irrad_var/istc * (nameplate_power + a*log(irrad_var/istc) + 
    # b*log(irrad_var/istc)^2 + c*(temp_var - tref) + d*(temp_var - tref)*log(irrad_var/istc) + 
    # e*(temp_var - tref)*log(irrad_var/istc)^2 + f*(temp_var - tref)^2).
    def plr_6k_model(self, df, var_list, by, nameplate_power, data_cutoff, predict_data):
        # initialize dataframes from helper function above
        model_df = self.model_initialization(df=df, var_list=var_list, by=by)

        # get predicted data from helper function above
        pred = self.generate_predicted_data(model_df=model_df, var_list=var_list, predict_data=predict_data)

        # determine number of data points for each time period
        n = model_df.groupby("tvar").size().reset_index(name="n")

        res_dfs = []
        split_var = 'tvar'

        # Iterate over each group in the DataFrame
        groups = model_df.groupby(split_var).groups
        grouped_dfs = [model_df.loc[group] for group in groups.values()]

        # Iterating variables
        start = 0
        end = n["n"][0]

        # Constants
        istc = 1000
        tref = pred['temp_var']

        def model_6k(x, c1, c2, c3, c4, c5, c6, name_power=nameplate_power):
            irrad_var, temp_var, tref = x
            istc = 1000  # Assuming istc is a constant value
            return irrad_var / istc * (name_power +
                                        c1 * np.log(irrad_var / istc) +
                                        c2 * np.log(irrad_var / istc) ** 2 +
                                        c3 * (temp_var - tref) +
                                        c4 * (temp_var - tref) * np.log(irrad_var / istc) +
                                        c5 * (temp_var - tref) * np.log(irrad_var / istc) ** 2 +
                                        c6 * (temp_var - tref) ** 2)

        for i, df in enumerate(grouped_dfs, start=0):
            mod = model_df[start:end]
            start = end
            end = end + n["n"][i]

            irrad_var = mod['irrad_var'].values
            temp_var = mod['temp_var'].values
            
            x_data = np.array([irrad_var, temp_var, np.full(mod['temp_var'].shape, tref[i])])
            power_var = mod['power_var'].values
            y_data = np.array(power_var)
            
            p0 = [0.6, 0.1, -0.05, -0.05, -0.005, 0.001]
            popt, pcov = curve_fit(model_6k, x_data, y_data, p0=p0, maxfev=200)
            
            c1, c2, c3, c4, c5, c6 = popt

            # Create predicted values array with the same size as mod array
            x_pred = np.array([pred['irrad_var'][i], pred['temp_var'][i], pred['temp_var'][i]])
            y_pred = np.zeros_like(y_data)
            for j in range(len(y_data)):
                y_pred[j] = model_6k(x_data[:, j], c1, c2, c3, c4, c5, c6)
            
            r_2 = r2_score(y_data, y_pred)
            adj_r2 = 1 - (1 - r_2) * (len(y_data) - 1) / (len(y_data) - len(popt) - 1)        
            mse = np.mean((y_data - y_pred) ** 2)
            
            res_df = pd.DataFrame({
                'tvar': mod['tvar'].iloc[0],
                'prediction': y_pred.mean(),
                'r_squared': r_2,
                'adj_r_squared': adj_r2,
                'mse': mse,
            }, index=[0])

            res_dfs.append(res_df)

        res_dfs = pd.concat(res_dfs, ignore_index=True)
        res_dfs = pd.merge(res_dfs, n, on='tvar', how='left')
        res_dfs['std_error'] = np.sqrt(res_dfs['mse']) / np.sqrt(res_dfs['n'])

        res_dfs = res_dfs[res_dfs['n'] >= data_cutoff]
        res_dfs['tvar'] = pd.to_numeric(res_dfs['tvar'])

        fitted_values = res_dfs['prediction']
        iqr = stats.iqr(fitted_values)
        lower = np.quantile(fitted_values, 0.25) - 1.5 * iqr
        upper = np.quantile(fitted_values, 0.75) + 1.5 * iqr

        res_dfs['outlier'] = res_dfs['prediction'].apply(lambda x: (x > upper) | (x < lower))

        res_dfs = res_dfs.assign(
            time_var=res_dfs['tvar'],
            power_var=res_dfs['prediction'],
            sigma=res_dfs['std_error'] * np.sqrt(res_dfs['n']),
        )[['time_var', 'power_var', 'std_error', 'sigma', 'outlier']]

        return res_dfs

'''
# DATA CLEANING ------------------------------------------------------
processor = PLRProcessor()
var_list = processor.plr_build_var_list(time_var='tmst', power_var='idcp', irrad_var='poay', temp_var='modt', wind_var='wspa')
RTC_DATA_PATH = "/home/ssk213/CSE_MSE_RXF131/vuv-data/proj/IEA-PVPS-T13/DOE-RTC-Baseline-datashare/c10hov6.csv"
dataf = pd.read_csv(RTC_DATA_PATH)
df = processor.plr_cleaning(df=dataf, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf = processor.plr_saturation_removal(df=df, var_list=var_list)

# DATA MODELING ------------------------------------------------------
model = PLRModel()
m = model.plr_pvusa_model(df=fdf, var_list=var_list, by='week', data_cutoff=30, predict_data=None)
model_no_outliers = processor.plr_remove_outlier(m)
print(model_no_outliers[0:60])

plt.scatter(model_no_outliers['time_var'], model_no_outliers['power_var'], color='grey')
plt.ylim(0, 4)
plt.grid(True)
plt.savefig('PVUSA_PLR.png')
'''

