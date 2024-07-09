""" Testing Module

This script is designed to test the modeling functions and compare calculated PLR values for 10 sets of data

"""

from feature_correction import PLRProcessor
from model_comparison import PLRModel
from plr_determination import PLRDetermination
import pandas as pd
import numpy as np

processor = PLRProcessor()
model = PLRModel()
det = PLRDetermination()

var_list = processor.plr_build_var_list(time_var='tmst', power_var='power1', irrad_var='poa', temp_var='Tcell', wind_var='wind_speed')
var_list2 = processor.plr_build_var_list(time_var='tmst', power_var='power2', irrad_var='poa', temp_var='Tcell', wind_var='wind_speed')



print("Test_36.1_-107.61 - power column 1 - 36.0989_-107.5809_-0.1386811------------------------------------------------")

dataf = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_36.1_-107.61.csv')
df = processor.plr_cleaning(df=dataf, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf = processor.plr_saturation_removal(df=df, var_list=var_list)

m = model.plr_xbx_utc_model(df=fdf, var_list=var_list, by='day', data_cutoff=0, predict_data=None)
no_outliers = processor.plr_remove_outlier(m)
print(no_outliers)

reg = det.plr_weighted_regression(data=no_outliers, power_var='power_var', time_var='time_var', model="Xbx-UTC", per_year=365, weight_var='sigma')
print(reg)
error = np.abs(reg['plr'].item() + 0.1386811)/0.1386811 * 100
print(f'The error of plr for this sample is {error:.2f}%')

# Calculated PLR = -0.10425



print("Test_36.1_-107.61 - power column 2 - 36.1153_-107.6025_-0.1190518------------------------------------------------")
dataf2 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_36.1_-107.61.csv')
df2 = processor.plr_cleaning(df=dataf2, var_list=var_list2, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf2 = processor.plr_saturation_removal(df=df2, var_list=var_list2)

m2 = model.plr_xbx_utc_model(df=fdf2, var_list=var_list2, by='day', data_cutoff=0, predict_data=None)
no_outliers2 = processor.plr_remove_outlier(m2)
print(no_outliers2)

reg2 = det.plr_weighted_regression(data=no_outliers2, power_var='power_var', time_var='time_var', model="Xbx-UTC", per_year=365, weight_var='sigma')
print(reg2)
error2 = np.abs(reg2['plr'].item() + 0.1190518)/0.1190518 * 100
print(f'The error of plr for this sample is {error2:.2f}%')

# Calculated PLR = -0.094143



print("Test_38.49_-108.38 - power column 1 - 38.4748_-108.3656_0.0107552------------------------------------------------")
dataf3 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_38.49_-108.38.csv')
df3 = processor.plr_cleaning(df=dataf3, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf3 = processor.plr_saturation_removal(df=df3, var_list=var_list)

m3 = model.plr_xbx_model(df=fdf3, var_list=var_list, by='month', data_cutoff=30, predict_data=None)
no_outliers3 = processor.plr_remove_outlier(m3)
print(no_outliers3)

reg3 = det.plr_weighted_regression(data=no_outliers3, power_var='power_var', time_var='time_var', model="Xbx", per_year=12, weight_var='sigma')
print(reg3)
error3 = np.abs(reg3['plr'].item() - 0.0107552)/0.0107552 * 100
print(f'The error of plr for this sample is {error3:.2f}%')

# Calculated PLR = -0.0044



print("Test_38.49_-108.38 - power column 2 - 38.4827_-108.3945_0.0089197------------------------------------------------")
dataf4 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_38.49_-108.38.csv')
df4= processor.plr_cleaning(df=dataf4, var_list=var_list2, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf4 = processor.plr_saturation_removal(df=df4, var_list=var_list2)

m4 = model.plr_xbx_model(df=fdf4, var_list=var_list2, by='month', data_cutoff=30, predict_data=None)
no_outliers4 = processor.plr_remove_outlier(m4)
print(no_outliers4)

reg4 = det.plr_weighted_regression(data=no_outliers4, power_var='power_var', time_var='time_var', model="Xbx", per_year=12, weight_var='sigma')
print(reg4)
error4 = np.abs(reg4['plr'].item() - 0.0089197)/0.0089197 * 100
print(f'The error of plr for this sample is {error4:.2f}%')

# Calculated PLR: -0.0589



print("Test_39.17_-103.68 -  power column 1 - 39.1576_-103.6065_-0.2969068------------------------------------------------")
dataf5 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_39.17_-103.68.csv')
df5= processor.plr_cleaning(df=dataf5, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf5 = processor.plr_saturation_removal(df=df5, var_list=var_list)

m5 = model.plr_xbx_model(df=fdf5, var_list=var_list, by='day', data_cutoff=0, predict_data=None)
no_outliers5 = processor.plr_remove_outlier(m5)
print(no_outliers5)

reg5 = det.plr_weighted_regression(data=no_outliers5, power_var='power_var', time_var='time_var', model="Xbx", per_year=365, weight_var='sigma')
print(reg5)
error5 = np.abs(reg5['plr'].item() + 0.2969068)/0.2969068 * 100
print(f'The error of plr for this sample is {error5:.2f}%')

# Calculated PLR = -0.348763



print("Test_39.17_-103.68 -  power column 2 - 39.1607_-103.7597_-0.2319716------------------------------------------------")
dataf6 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_39.17_-103.68.csv')
df6 = processor.plr_cleaning(df=dataf6, var_list=var_list2, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf6 = processor.plr_saturation_removal(df=df6, var_list=var_list2)

m6 = model.plr_pvusa_model(df=fdf6, var_list=var_list2, by='day', data_cutoff=0, predict_data=None)
no_outliers6 = processor.plr_remove_outlier(m6)
print(no_outliers6)

reg6 = det.plr_weighted_regression(data=no_outliers6, power_var='power_var', time_var='time_var', model="PUVSA", per_year=365, weight_var='sigma')
print(reg6)
error6 = np.abs(reg6['plr'].item() + 0.2319716)/0.2319716 * 100
print(f'The error of plr for this sample is {error6:.2f}%')

# Calculated PLR: -0.1133



print("Test_39.74_-107.82 -  power column 1 - 39.723_-107.7729_0.063436------------------------------------------------")
dataf7 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_39.74_-107.82.csv')
df7 = processor.plr_cleaning(df=dataf7, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf7 = processor.plr_saturation_removal(df=df7, var_list=var_list)

m7 = model.plr_xbx_model(df=fdf7, var_list=var_list, by='month', data_cutoff=30, predict_data=None)
no_outliers7 = processor.plr_remove_outlier(m7)
print(no_outliers7)

reg7 = det.plr_weighted_regression(data=no_outliers7, power_var='power_var', time_var='time_var', model="Xbx", per_year=12, weight_var='sigma')
print(reg7)
error7 = np.abs(reg7['plr'].item() - 0.063436)/0.063436 * 100
print(f'The error of plr for this sample is {error7:.2f}%')

# Calculated PLR: 0.0103



print("Test_39.74_-107.82 -  power column 2 - 39.7246_-107.8476_0.0481901------------------------------------------------")
dataf8 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_39.74_-107.82.csv')
df8 = processor.plr_cleaning(df=dataf8, var_list=var_list2, irrad_thresh=200, low_power_thresh=0.05, high_power_cutoff=None)
fdf8 = processor.plr_saturation_removal(df=df8, var_list=var_list2)

m8 = model.plr_xbx_model(df=fdf8, var_list=var_list2, by='month', data_cutoff=0, predict_data=None)
no_outliers8 = processor.plr_remove_outlier(m8)
print(no_outliers8)

reg8 = det.plr_weighted_regression(data=no_outliers8, power_var='power_var', time_var='time_var', model="Xbx", per_year=12, weight_var='sigma')
print(reg8)
error8 = np.abs(reg8['plr'].item() - 0.0481901)/0.0481901 * 100
print(f'The error of plr for this sample is {error8:.2f}%')

# Calculated PLR: 0.050879




print("Test_39.86_-107.43 -  power column 1 - 39.8622_-107.4302_0.021062------------------------------------------------")
dataf9 = pd.read_csv('/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git/pvplr-suraj-2/pvplr/testing/Test_39.86_-107.43.csv')
df9 = processor.plr_cleaning(df=dataf9, var_list=var_list, irrad_thresh=800, low_power_thresh=0.05, high_power_cutoff=None)
fdf9 = processor.plr_saturation_removal(df=df9, var_list=var_list)

m9 = model.plr_xbx_model(df=fdf9, var_list=var_list, by='month', data_cutoff=30, predict_data=None)
no_outliers9 = processor.plr_remove_outlier(m9)
print(no_outliers9)

reg9 = det.plr_weighted_regression(data=no_outliers9, power_var='power_var', time_var='time_var', model="Xbx", per_year=12, weight_var='sigma')
print(reg9)
error9 = np.abs(reg9['plr'].item() - 0.021062)/0.021062 * 100
print(f'The error of plr for this sample is {error9:.2f}%')

# Calculated PLR: 0.0164
