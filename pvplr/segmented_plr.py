from feature_correction import PLRProcessor
from model_comparison import PLRModel
import piecewise_regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plr_seg_extract(df, n_breakpoints, per_year, power_var, time_var, return_model = False):
    """
    Perform piecewise linear regression on the given data.

    Parameters:
    df (pd.DataFrame): Data frame of corrected power measurements
    n_breakpoints (int): Number of desired breakpoints
    per_year (int): 365 for daily, 52 for weekly, 12 for monthly
    power_var (str): Name of the power variable column
    time_var (str): Name of the time variable column
    return_model (bool): If True, return model summary stats; if False, return PLR results

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

    '''
    # Plotting
    pw_fit.plot_data(color="grey", s=20)    
    pw_fit.plot_fit(color="red", linewidth=1)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.ylim(2000,2400)
    plt.show()
    plt.savefig('Test_Seg_PLR_W.png')
    '''

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

    return segments_data

# DATA CLEANING ------------------------------------------------------
processor = PLRProcessor()
var_list = processor.plr_build_var_list(time_var='timestamp', power_var='power', irrad_var='g_poa', temp_var='mod_temp', wind_var='air_temp')
dataf = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr/data/2008sdle_plr_reduced_dataset.csv')
df = processor.plr_cleaning(df=dataf, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf = processor.plr_saturation_removal(df=df, var_list=var_list)
#print(fdf)

# DATA MODELING ------------------------------------------------------
model = PLRModel()
predict_data = pd.DataFrame({'irrad_var': [800], 'temp_var': [40], 'wind_var': [0]})  # use for Xbx only
m = model.plr_xbx_model(df=fdf, var_list=var_list, by='week', data_cutoff=10, predict_data=predict_data) 
#print(m)

stl_data = processor.plr_decomposition(data=m, by='W', freq=4, start_date="2015-11-24", power_var='power_var', time_var='time_var', 
                                        plot=False, plot_file='/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/Xbx_Decomposed_W', 
                                        title='Xbx Daily Decomposed PLR')
print(stl_data)

seg = plr_seg_extract(df=stl_data, n_breakpoints=1, per_year=365, power_var='power', time_var='age', return_model=False)
print(seg)
