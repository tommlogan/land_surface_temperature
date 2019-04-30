'''
Here are a collection of functions for analyzing the land surface temperature and
thermal radiance from LandSat images of cities
'''

# import libraries
import matplotlib as mpl
# mpl.use('Agg')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import joypy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import code
from joblib import Parallel, delayed
pd.options.mode.chained_assignment = 'raise'
import itertools
import glob
from math import *

# regression libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import mean_squared_error, r2_score
from pyearth import Earth
from pygam import LinearGAM

# init logging
import sys
# sys.path.append("code")
# from logger_config import *
# logger = logging.getLogger(__name__)

RANDOM_SEED = 3201

# other constants
city_names = {'bal':'baltimore','por':'portland','phx':'phoenix','det':'detroit'}
model_names = {'rf':'random forest','mlr':'multivariate linear',
                'gam':'generalized additive\n(gam)',
                'gbrt':'gradient boosted\ntrees',
                'mars':'multivariate adaptive\nspline (mars)',
                'cnn':'convolutional\nneural network'}
feature_names = {'lcov_11' : '% water','tree_mean':'% tree canopy','ndvi_mean':'ndvi',
                'svf_mean':'sky view factor','dsm_mean':'digital surface model',
                'alb_mean':'albedo','dsm_sd':'dsm stand. dev.','nbdi_max':'max nbdi',
                'tree_max':'max % tree can.','bldg':'% building area','pdens_mean':'pop. density',
                'tree_min':'min % tree can.','svf_max':'max sky view factor',
                'tree_sd': '% tree can. stand. dev.', 'nbdi_sd_sl':'nbdi surrounding stand. dev.',
                'tree_sd_sl':'% tree can. surrounding stand. dev.',
                'ndvi_sd': 'ndvi stand. dev'
                }

CORES_NUM = min(50,int(os.cpu_count()*3/4))

def main():
    '''
    '''
    # loop cities and all
    cities = ['bal', 'por', 'det', 'phx']
    # import data
    df = import_data(cities, grid_size)

    # present the data - density plot
    # plot_density(df, cities)

    # regression
    ## train on three cities, test on one
    loss = regression_cityholdouts(df, cities)
    # plot the points
    plot_holdout_points(loss)

    ## for each city, train and test
    sim_num = 50 # number of holdouts
    regressions(df, cities, sim_num)
    plot_holdouts()

    # variable importance and partial dependence
    # reg_gbm = full_gbm_regression(df, cities)

    # variable selection
    loop_variable_selection(df, cities)

    # based on the results of the variable selection, rerun the regression and
    # create the variable importance plots
    #vars_selected = ['tree_mean', 'ndvi_mean_mean', 'alb_mean_mean', 'elev_min_sl', 'elev_max', 'tree_max_sl']
    # vars_selected = ['tree_mean', 'ndvi_mean_mean', 'alb_mean_mean', 'alb_mean_min', 'elev_min_sl', 'ndvi_mean_min']
    # vars_selected = ['tree_mean', 'ndvi_mean_mean', 'alb_mean_mean', 'elev_min', 'alb_mean_min_sl', 'elev_max']
    # reg_gbm, X_train = full_gbm_regression(df, cities, vars_selected)
    # #
    # # # plot the variable importance
    # importance_order = plot_importance(reg_gbm, cities)
    # #
    # # # plot the partial dependence
    # plot_dependence(importance_order, reg_gbm, cities, X_train, vars_selected, show_plot=False)

def import_data(grid_size, selected_vars = True):
    if selected_vars:
        df = pd.read_csv('data/data_vif_{}.csv'.format(grid_size))
        df = df.drop('Unnamed: 0', axis=1)
    else:
        df = pd.read_csv('data/data_regressions_{}_20190419.csv'.format(grid_size))
        df = df.drop('Unnamed: 0', axis=1)
    return(df)


def regressions(df, cities, sim_num, grid_size, do_par = False):
    '''
    to compare regression out-of-bag accuracy I need to split into test and train
    I also want to scale some of the variables
    '''
    # for city in cities:
    df_city = df#[df['city']==city] #df_city = df.loc[df['city']==city]
    predict_quant = 'lst'
    df_city, response = prepare_lst_prediction(df_city)
    # conduct the holdout
    if do_par:
        Parallel(n_jobs=CORES_NUM)(delayed(single_regression)(df_city, response, grid_size, predict_quant, i) for i in range(sim_num))
    else:
        for i in range(sim_num):
            single_regression(df_city, response, grid_size, predict_quant, i)


def single_regression(df_city, response, grid_size, predict_quant, i):
    '''
    fit the different models for a single holdout
    '''
    city = str(i)
    # prepare the results of the holdout
    loss = pd.DataFrame()
    # divide into test and training sets
    X_train, X_test, y_train, y_test = split_holdout(df_city, response, test_size=0.20)#, random_state=RANDOM_SEED)
    # drop unnecessary variables
    X_train, X_test = subset_regression_data(X_train, X_test)
    # response values
    y = define_response_lst(y_train, y_test)
    # null model
    loss = regression_null(y, city, predict_quant, loss)
    # GradientBoostingRegressor
    loss = regression_gradientboost(X_train, y, X_test, city, predict_quant, loss)
    # multiple linear regression
    loss = regression_linear(X_train, y, X_test, city, predict_quant, loss)
    # random forest regression
    loss = regression_randomforest(X_train, y, X_test, city, predict_quant, loss)
    # mars
    loss = regression_mars(X_train, y, X_test, city, predict_quant, loss)
    # gam
    loss = regression_gam(X_train, y, X_test, city, predict_quant, loss)
    # save results
    loss.to_csv('data/regression/holdout_{}/holdout{}_results_{}.csv'.format(grid_size, i, grid_size))


def regression_cityholdouts(df, cities):
    '''
    to compare regression out-of-bag accuracy I need to split into test and train
    I also want to scale some of the variables
    '''
    predict_quant = 'lst'

    # prep y
    df, response = prepare_lst_prediction(df)
    loss = pd.DataFrame()
    for city in cities:
        train_idx = np.where(df['city'] != city)
        test_idx = np.where(df['city'] == city)
        # divide into test and training sets
        X_train = df.iloc[train_idx].copy()
        y_train = response.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()
        y_test = response.iloc[test_idx].copy()
        # drop unnecessary variables
        X_train, X_test = subset_regression_data(X_train, X_test)
        # response values
        y = define_response_lst(y_train, y_test)
        ### do the holdouts
        city = 'hold-{}'.format(city)
        # null model
        loss = regression_null(y, city, predict_quant, loss)
        # GradientBoostingRegressor
        loss = regression_gradientboost(X_train, y, X_test, city, predict_quant, loss)
        # multiple linear regression
        loss = regression_linear(X_train, y, X_test, city, predict_quant, loss)
        # random forest regression
        loss = regression_randomforest(X_train, y, X_test, city, predict_quant, loss)
        # mars
        loss = regression_mars(X_train, y, X_test, city, predict_quant, loss)
        # gam
        loss = regression_gam(X_train, y, X_test, city, predict_quant, loss)
    return(loss)


def prepare_lst_prediction(df):
    '''
    to predict for thermal radiance, let's remove land surface temp and superfluous
    thermal radiance values
    '''
    # drop lst
    lst_vars = ['lst_day_mean','lst_night_mean','lst_night_max','lst_day_max']
    lst_mean = df[lst_vars].copy()
    df = df.drop(lst_vars, axis=1)

    return(df, lst_mean)


def subset_regression_data(X_train, X_test):
    '''
    drop unnecessary variables
    '''
    vars_all = X_train.columns.values
    cities = np.unique(X_train['city'])

    # drop the following variables
    vars_drop = ['city','holdout','x','y']
    X_train = X_train.drop(vars_drop, axis=1)
    X_test = X_test.drop(vars_drop, axis=1)

    return(X_train, X_test)


def define_response_lst(y_train, y_test):
    y = {}
    y['day_train'] = y_train['lst_day_mean']
    y['night_train'] = y_train['lst_night_mean']
    y['nightmax_train'] = y_train['lst_night_max']
    y['daymax_train'] = y_train['lst_day_max']
    # test
    y['day_test'] = y_test['lst_day_mean']
    y['night_test'] = y_test['lst_night_mean']
    y['nightmax_test'] = y_test['lst_night_max']
    y['daymax_test'] = y_test['lst_day_max']
    return(y)


def calculate_partial_dependence(df, grid_size, boot_index = None):
    '''
    fit the models to the entire dataset
    loop through each feature
    vary the feature over its range
    predict the target variable to see how it is influenced by the feature
    '''
    results_partial = pd.DataFrame()
    df, target = prepare_lst_prediction(df)
    df  = subset_regression_data(df, df)[0]
    df_reference = df.copy()
    feature_resolution = 25
    # loop day and night
    for h in ['lst_day_mean', 'lst_night_mean', 'lst_night_max','lst_day_max']:
        print(h)
        ###
        # fit models
        ###
        # gradient boosted tree
        gbm = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
        gbm.fit(df, target[h])
        # random forest
        rf = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_features=1/3)
        rf.fit(df, target[h])
        # mars
        mars = Earth(max_degree=1, penalty=1.0, endspan=5)
        mars.fit(df, target[h])
        # GAM
        gam = LinearGAM(n_splines=10).fit(df, target[h])
        # linear
        mlr = LinearRegression()
        mlr = mlr.fit(df, target[h])
        ###
        # loop through features and their ranges
        ###
        for var_interest in list(df): #['tree_mean','density_housesarea']:
            # loop through range of var_interest
            var_values = np.linspace(np.percentile(df[var_interest],2.5),np.percentile(df[var_interest],97.5), feature_resolution)
            df_change = df.copy()
            for x in var_values:
                df_change[var_interest] = x
                # gbm
                pred = gbm.predict(df_change)
                # save results
                results_partial = results_partial.append({'model': 'gbrt', 'dependent':h,'independent':var_interest,
                                                          'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                # rf
                pred = rf.predict(df_change)
                # save results
                results_partial = results_partial.append({'model': 'rf', 'dependent':h,'independent':var_interest,
                                                          'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                # mars
                pred = mars.predict(df_change)
                # save results
                results_partial = results_partial.append({'model': 'mars', 'dependent':h,'independent':var_interest,
                                                          'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                # gam
                pred = gam.predict(df_change)
                # save results
                results_partial = results_partial.append({'model': 'gam', 'dependent':h,'independent':var_interest,
                                                          'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                # mlr
                pred = mlr.predict(df_change)
                # save results
                results_partial = results_partial.append({'model': 'mlr', 'dependent':h,'independent':var_interest,
                                                          'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
            # save results
            if boot_index:
                results_partial.to_csv('data/regression/bootstrap_{}/results_partial_dependence_{}.csv'.format(grid_size,boot_index))
            else:
                results_partial.to_csv('data/regression/results_partial_dependence_{}.csv'.format(grid_size))


def calc_swing(results_pd, grid_size):
    '''
    calculate the variable importance (swing)
    input: the results of the partial dependence
    now calculate the maximum change of the target for each feature
    the result is the swing, a measure of variable importance
    '''
    features = np.unique(results_pd['independent'])
    targets = np.unique(results_pd['dependent'])
    models = np.unique(results_pd['model'])
    # init df
    results_swing = pd.DataFrame()
    # loop targets
    for h in targets:
        # loop models
        for m in models:
            # calculate the range of the target for each feature
            model_range = results_pd.loc[(results_pd['dependent']==h) & (results_pd['model']==m)].groupby(['independent']).agg(np.ptp)
            # calc sum of range
            range_sum = model_range['mean'].sum()
            # calc swing
            swing = model_range['mean']/range_sum
            # put into dataframe
            swing = swing.to_frame('swing').reset_index()
            swing['model'] = m
            swing['dependent'] = h
            swing['raw'] = model_range['mean'].values
            # save
            results_swing = results_swing.append(swing, ignore_index=True)
        # save results
        results_swing.to_csv('data/regression/results_swing_{}.csv'.format(grid_size))


def bootstrap_main(df, grid_size, boot_num, do_par = False):
    '''
    loop the bootstraps to calculate the partial_dependence
    '''
    if do_par:
        Parallel(n_jobs=CORES_NUM)(delayed(boot_pd)(df, grid_size, boot_index) for boot_index in range(boot_num))
    else:
        for boot_index in range(boot_num):
            boot_pd(df, grid_size, boot_index)
    # import the data
    path = 'data/regression/bootstrap_{}'.format(grid_size)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    results_pd = pd.concat(df_from_each_file, ignore_index=True)
    # undo the normalization
    normalize_parameters = pd.read_csv('data/normalization_parameters_{}.csv'.format(grid_size))
    normalize_parameters = normalize_parameters.set_index('feature')
    for index, row in results_pd.iterrows():
        feature = row.independent
        results_pd.loc[index,'x'] = row['x']*normalize_parameters.loc[feature,'sd'] + normalize_parameters.loc[feature,'mean']

    # save results
    results_pd.to_csv('data/regression/results_partial_dependents_{}.csv'.format(grid_size))


def boot_pd(df, grid_size, boot_index):
    '''
    sample the holdout numbers with replacement to create bootstrapped df
    fit the models and calculate the partial dependence
    '''
    # resample the holdout numbers
    holdout_numbers = np.unique(df.holdout)
    holdout_sample = np.random.choice(holdout_numbers, len(holdout_numbers), replace=True)
    # create the df based on this sample
    sample_index = [list(df[df.holdout == x].index) for x in holdout_sample]
    sample_index = list(itertools.chain.from_iterable(sample_index))
    df = df.loc[sample_index]
    # calculate the pd
    calculate_partial_dependence(df, grid_size, boot_index)


def calculate_partial_dependence_city(df_full, grid_size, cities):
    '''
    fit the models to the entire dataset
    loop through each feature
    vary the feature over its range
    predict the target variable to see how it is influenced by the feature
    '''
    results_partial = pd.DataFrame()
    for city in cities:
        print(city)
        df = df_full[df_full.city == city]
        df, target = prepare_lst_prediction(df)
        df = subset_regression_data(df, df)[0]
        df_reference = df.copy()
        feature_resolution = 25
        # loop day and night
        for h in ['lst_day_mean', 'lst_night_mean', 'lst_night_max','lst_day_max']:
            print(h)
            ###
            # fit models
            ###
            # gradient boosted tree
            # gbm = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
            # gbm.fit(df, target[h])
            # random forest
            rf = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_features=1/3)
            rf.fit(df, target[h])
            # mars
            # mars = Earth(max_degree=1, penalty=1.0, endspan=5)
            # mars.fit(df, target[h])
            # GAM
            # gam = LinearGAM(n_splines=10).fit(df, target[h])
            # linear
            # mlr = LinearRegression()
            # mlr = mlr.fit(df, target[h])
            ###
            # loop through features and their ranges
            ###
            if grid_size == 500:
                var_interests = {'lst_day_mean':['lcov_11', 'ndvi_mean', 'tree_mean', 'alb_mean','dsm_mean'],
                                'lst_night_mean':['tree_mean', 'ndvi_mean', 'lcov_11', 'svf_mean', 'dsm_mean'],
                                'lst_day_max':['tree_min', 'ndvi_mean','alb_mean','lcov_11','nbdi_max'],
                                'lst_night_max':['tree_min', 'ndvi_mean','tree_mean','dsm_mean','svf_mean']}
            else:
                var_interests = {'lst_day_mean':['tree_mean', 'ndvi_mean', 'alb_mean','dsm_mean','svf_mean'],
                                'lst_night_mean':['tree_mean', 'ndvi_mean', 'alb_mean', 'svf_mean', 'dsm_mean'],
                                'lst_day_max':['ndvi_mean','tree_mean','alb_mean','tree_sd','dsm_mean'],
                                'lst_night_max':['tree_mean', 'ndvi_mean','tree_sd','svf_mean','alb_mean']}
            for var_interest in var_interests[h]: #['tree_mean','density_housesarea']:
                # loop through range of var_interest
                var_values = np.linspace(np.percentile(df[var_interest],2.5),np.percentile(df[var_interest],97.5), feature_resolution)
                df_change = df.copy()
                for x in var_values:
                    df_change[var_interest] = x
                    # gbm
                    # pred = gbm.predict(df_change)
                    # # save results
                    # results_partial = results_partial.append({'model': 'gbrt', 'dependent':h,'independent':var_interest,
                                                              # 'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                    # rf
                    pred = rf.predict(df_change)
                    # save results
                    results_partial = results_partial.append({'model': 'rf', 'dependent':h,'independent':var_interest,
                                                              'x':x, 'mean':np.mean(pred), 'city': city}, ignore_index=True)
                    # mars
                    # pred = mars.predict(df_change)
                    # # save results
                    # results_partial = results_partial.append({'model': 'mars', 'dependent':h,'independent':var_interest,
                    #                                           'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                    # # gam
                    # pred = gam.predict(df_change)
                    # # save results
                    # results_partial = results_partial.append({'model': 'gam', 'dependent':h,'independent':var_interest,
                    #                                           'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                    # # mlr
                    # pred = mlr.predict(df_change)
                    # # save results
                    # results_partial = results_partial.append({'model': 'mlr', 'dependent':h,'independent':var_interest,
                                                              # 'x':x, 'mean':np.mean(pred), 'boot': boot_index}, ignore_index=True)
                # save results
                results_partial.to_csv('data/regression/city_{}/results_partial_dependence_{}.csv'.format(grid_size, grid_size))


###
# Regression code
###

def regression_null(y, city, predict_quant, loss):
    '''
    fit the null model for comparison
    '''
    # train the model
    model = 'average'

    # predict the model
    predict_day = np.ones(len(y['day_test'])) * np.mean(y['day_train'])
    predict_night = np.ones(len(y['night_test'])) * np.mean(y['night_train'])
    predict_nightmax = np.ones(len(y['nightmax_test'])) * np.mean(y['nightmax_train'])
    predict_daymax = np.ones(len(y['daymax_test'])) * np.mean(y['daymax_train'])

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'null', city, predict_quant)

    # calculate the MAE
    mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax = calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax)

    # record results
    loss = record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax)

    return(loss)

def regression_gradientboost(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'gbrf'
    # train the model
    gbm_day_reg = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
    gbm_night_reg = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
    reg_nightmax = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
    reg_daymax = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
    # code.interact(local = locals())
    gbm_day_reg.fit(X_train, y['day_train'])
    gbm_night_reg.fit(X_train, y['night_train'])
    reg_daymax.fit(X_train, y['daymax_train'])
    reg_nightmax.fit(X_train, y['nightmax_train'])

    # predict the model
    predict_day = gbm_day_reg.predict(X_test)
    predict_night = gbm_night_reg.predict(X_test)
    predict_daymax = reg_daymax.predict(X_test)
    predict_nightmax = reg_nightmax.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax = calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax)


    # record results
    loss = record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax)

    return(loss)

def regression_linear(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the multiple linear regressions
    '''
    model = 'mlr'
    # train the model
    mlr_day_reg = LinearRegression()
    mlr_night_reg = LinearRegression()
    reg_daymax = LinearRegression()
    reg_nightmax = LinearRegression()

    mlr_day_reg.fit(X_train, y['day_train'])
    mlr_night_reg.fit(X_train, y['night_train'])
    reg_daymax.fit(X_train, y['daymax_train'])
    reg_nightmax.fit(X_train, y['nightmax_train'])


    # predict the model
    predict_day = mlr_day_reg.predict(X_test)
    predict_night = mlr_night_reg.predict(X_test)
    predict_daymax = reg_daymax.predict(X_test)
    predict_nightmax = reg_nightmax.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'mlr', city, predict_quant)

    # calculate the MAE
    mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax = calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax)

    # record results
    loss = record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax)

    return(loss)

def regression_randomforest(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'rf'
    # train the model
    reg_day = RandomForestRegressor(n_estimators=500, max_features=1/3)
    reg_night = RandomForestRegressor(n_estimators=500, max_features=1/3)
    reg_nightmax = RandomForestRegressor(n_estimators=500, max_features=1/3)
    reg_daymax = RandomForestRegressor(n_estimators=500, max_features=1/3)
    reg_day.fit(X_train, y['day_train'])
    reg_night.fit(X_train, y['night_train'])
    reg_daymax.fit(X_train, y['daymax_train'])
    reg_nightmax.fit(X_train, y['nightmax_train'])

    # predict the model
    predict_day = reg_day.predict(X_test)
    predict_night = reg_night.predict(X_test)
    predict_daymax = reg_daymax.predict(X_test)
    predict_nightmax = reg_nightmax.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax = calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax)

    # record results
    loss = record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax)

    return(loss)

def regression_mars(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'mars'
    # train the model
    reg_day = Earth(max_degree=1, penalty=1.0, endspan=5)
    reg_night = Earth(max_degree=1, penalty=1.0, endspan=5)
    reg_nightmax = Earth(max_degree=1, penalty=1.0, endspan=5)
    reg_daymax = Earth(max_degree=1, penalty=1.0, endspan=5)
    reg_day.fit(X_train, y['day_train'])
    reg_night.fit(X_train, y['night_train'])
    reg_daymax.fit(X_train, y['daymax_train'])
    reg_nightmax.fit(X_train, y['nightmax_train'])

    # predict the model
    predict_day = reg_day.predict(X_test)
    predict_night = reg_night.predict(X_test)
    predict_daymax = reg_daymax.predict(X_test)
    predict_nightmax = reg_nightmax.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax = calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax)

    # record results
    loss = record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax)

    return(loss)

def regression_gam(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'gam'
    # train the model
    reg_day = LinearGAM(n_splines=10)
    reg_night = LinearGAM(n_splines=10)
    reg_nightmax = LinearGAM(n_splines=10)
    reg_daymax = LinearGAM(n_splines=10)
    reg_day.fit(X_train, y['day_train'])
    reg_night.fit(X_train, y['night_train'])
    reg_daymax.fit(X_train, y['daymax_train'])
    reg_nightmax.fit(X_train, y['nightmax_train'])

    # predict the model
    predict_day = reg_day.predict(X_test)
    predict_night = reg_night.predict(X_test)
    predict_daymax = reg_daymax.predict(X_test)
    predict_nightmax = reg_nightmax.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax = calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax)

    # record results
    loss = record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax)

    return(loss)

def full_gbm_regression(df, cities, vars_selected=None):
    '''
    fit gbm on the entire dataset and return the objects
    '''
    if vars_selected is None:
        vars_selected = []
    reg_gbm = {}
    reg_gbm['diurnal'] = {}
    reg_gbm['nocturnal'] = {}
    X_train = {}
    predict_quant = 'lst'
    cities = cities.copy()
    cities.append('all')
    for city in cities:
        # subset for the city
        if city != 'all':
            df_city = df[df['city']==city].copy()
        else:
            df_city = df.copy()
        # drop necessary variables
        df_city, response = prepare_lst_prediction(df_city)
        # keep only specified variables, if any were specified
        if len(vars_selected)>0:
            df_city = df_city[vars_selected+['city']]
        # no need to divide, but split into X and y
        X_train[city], X_test, y_train, y_test = split_holdout(df_city, response, test_size=0)#, random_state=RANDOM_SEED)
        print(len(X_train[city]), len(X_test))
        # drop unnecessary variables
        X_train, X_test = subset_regression_data(X_train, X_test)
        # response values
        y = define_response_lst(y_train, y_train)
        # fit the model
        reg_gbm['diurnal'][city] = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
        reg_gbm['diurnal'][city].fit(X_train[city], y['day_train'])
        reg_gbm['nocturnal'][city] = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
        reg_gbm['nocturnal'][city].fit(X_train[city], y['night_train'])
    reg_gbm['covariates'] = X_train[city].columns
    return(reg_gbm, X_train)

def record_result(loss, city, model, mae_day, mae_night, r2_night, r2_day, mae_daymax, mae_nightmax, r2_nightmax, r2_daymax):
    loss = loss.append({
        'time_of_day': 'diurnal',
        'hold_num': city,
        'model': model,
        'error_metric': 'r2',
        'error': r2_day
    }, ignore_index=True)
    loss = loss.append({'time_of_day': 'diurnal','hold_num': city,'model': model,'error_metric': 'mae','error': mae_day}, ignore_index=True)
    loss = loss.append({'time_of_day': 'nocturnal','hold_num': city,'model': model,'error_metric': 'mae','error': mae_night}, ignore_index=True)
    loss = loss.append({'time_of_day': 'nocturnal','hold_num': city,'model': model,'error_metric': 'r2','error': r2_night}, ignore_index=True)
    loss = loss.append({'time_of_day': 'nocturnalmax','hold_num': city,'model': model,'error_metric': 'r2','error': r2_nightmax}, ignore_index=True)
    loss = loss.append({'time_of_day': 'diurnalmax','hold_num': city,'model': model,'error_metric': 'r2','error': r2_daymax}, ignore_index=True)
    loss = loss.append({'time_of_day': 'nocturnalmax','hold_num': city,'model': model,'error_metric': 'mae','error': mae_nightmax}, ignore_index=True)
    loss = loss.append({'time_of_day': 'diurnalmax','hold_num': city,'model': model,'error_metric': 'mae','error': mae_daymax}, ignore_index=True)

    return(loss)

def calculate_errors(y, predict_day, predict_night, predict_daymax, predict_nightmax):
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)
    mae_daymax = np.mean(abs(predict_daymax - y['daymax_test']))
    mae_nightmax = np.mean(abs(predict_nightmax - y['nightmax_test']))
    r2_daymax = r2_score(y['daymax_test'], predict_daymax)
    r2_nightmax = r2_score(y['nightmax_test'], predict_nightmax)
    return(mae_day, mae_night,r2_day,r2_night,mae_daymax,mae_nightmax,r2_daymax,r2_nightmax)

###
# Supporting code
###

def split_holdout(df, response, test_size):
    '''
    Prepare spatial holdout
    '''
    # what is the total number of records?
    n_records = df.shape[0]
    # what are the holdout numbers to draw from?
    holdout_freq = df.groupby('holdout')['holdout'].count()
    holdout_options = list(holdout_freq.index)
    # required number of records
    req_records = n_records * (test_size*0.95)
    # select holdout groups until required number of records is achieved
    heldout_records = 0
    heldout_groups = []
    while heldout_records < req_records:
        # randomly select a holdout group to holdout
        hold = np.random.choice(holdout_options, 1, replace = False)[0]
        # remove that from the options
        holdout_options.remove(hold)
        # add that to the heldout list
        heldout_groups.append(hold)
        # calculate the number of records held out
        heldout_records = holdout_freq.loc[heldout_groups].sum()
    # create the test and training sets
    X_test = df[df.holdout.isin(heldout_groups)]
    y_test = response[df.holdout.isin(heldout_groups)]
    X_train = df[~df.holdout.isin(heldout_groups)]
    y_train = response[~df.holdout.isin(heldout_groups)]
    return(X_train, X_test, y_train, y_test)

def loop_variable_selection(df, cities):
    from datetime import datetime
    vars_forward = {}
    vars_forward['day'] = {}
    vars_forward['night'] = {}
    for city in ['all'] + cities:
        for period in ['day','night']:
            print('{}: Starting {}, {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), city, period))
            vars_forward[period][city] = feature_selection(25, city, df, period)
            print('{}: Completed {}, {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), city, period))
    with open('data/variable_selection.pkl', 'wb') as f:
        pickle.dump(vars_forward, f, pickle.HIGHEST_PROTOCOL)

def feature_selection(holdout_num, city, df, period):
    '''
    forward selection of variables based on OOB mae
    '''
    df_set, response = prepare_lst_prediction(df)
    variables = df_set.columns.values
    variables = [var for var in variables if var not in ['city','area']]
    # subset for the city
    if city != 'all':
        df_city = df[df['city']==city].copy()
    else:
        df_city = df.copy()
    # drop necessary variables
    df_city, response = prepare_lst_prediction(df_city)
    # add variables based on which provide the best improvement to lowering MAE
    vars_inc = []
    vars_mae = []
    num_vars = len(variables)
    while len(vars_inc)<num_vars:
        # loop through the Variables
        variables = [var for var in variables if var not in vars_inc]
        variable_mae = pd.DataFrame(index=variables, columns=['mae'])
        for var in variables:
            df_var = df_city.loc[:,[var,'city'] + vars_inc].copy()
            # initialize error measures
            mae = []
            for h in range(holdout_num):
                # no need to divide, but split into X and y
                X_train, X_test, y_train, y_test = split_holdout(df_var, response, test_size=0.25)#, random_state=RANDOM_SEED)
                # drop unnecessary variables
                X_train, X_test = subset_regression_data(X_train.copy(), X_test.copy())
                # response values
                y = define_response_lst(y_train, y_test)
                # fit the model
                gbm_day = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
                gbm_day.fit(X_train, y['{}_train'.format(period)])
                # predict the model
                predict_day = gbm_day.predict(X_test)
                # calculate MAE
                mae.append(np.mean(abs(predict_day - y['{}_test'.format(period)])))
            # calculate the average
            variable_mae.loc[var,'mae'] = np.mean(mae)
        # variable to include
        vars_inc.append(variables[variable_mae.loc[:,'mae'].values.argmin()])
        vars_mae.append(variable_mae.loc[:,'mae'].values.min())
    # add to dict
    var_forwardstep = pd.DataFrame({
        'variables':vars_inc,
        'mae':vars_mae
    })
    return(var_forwardstep)


###
# Plotting code
###
# plt.style.use(['seaborn-colorblind'])#,'dark_background'])
plt.style.use(['tableau-colorblind10'])#,'dark_background'])
fig_transparency = False
# figure size (cm)
width_1col = 8.7/2.54
width_2col = 17.8/2.54
golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
height_1c = width_1col*golden_mean
height_2c = width_2col*golden_mean
aspect_2c = width_2col/height_2c
# font size
font_size = 11
dpi = 500
# additional parameters
params = {'axes.labelsize': font_size, # fontsize for x and y labels (was 10)
          'font.size': font_size, # was 10
          'legend.fontsize': font_size * 2/3, # was 10
          'xtick.labelsize': font_size,
          'font.sans-serif' : 'Corbel',
          # 'ytick.labelsize': 0,
          'lines.linewidth' : 1,
          'figure.autolayout' : True,
          'figure.figsize': [width_2col, height_2c],#[fig_width/2.54,fig_height/2.54]
          'axes.spines.top'    : False,
          'axes.spines.right'  : False,
          'axes.xmargin' : 0
}
mpl.rcParams.update(params)



def plot_density(df, cities):
    '''
    output density plots of the variables
    '''
    for city in cities:
        # logger.info('density plotting for {}'.format(city))
        df_city = df.loc[df['city']==city]
        # thermal radiance
        df_city['tr_day_mean'].plot(kind='density', label = "diurnal", alpha = 0.5)
        df_city['tr_nght_mean'].plot(kind='density', label = "nocturnal", alpha = 0.5)
        plt.legend(loc='upper right')
        plt.title("Mean {} Thermal Radiance".format(city))
        plt.xlabel('Thermal Radiance [unit?]')
        plt.savefig('fig/working/density/therm-rad_{}.pdf'.format(city), format='pdf', dpi=1000, transparent=True)
        plt.clf()
        #
        # land surface temperature
        df_city['lst_day_mean_mean'].plot(kind='density', label = "diurnal", alpha = 0.5)
        df_city['lst_night_mean_mean'].plot(kind='density', label = "nocturnal", alpha = 0.5)
        plt.legend(loc='upper right')
        plt.title("Mean {} Land Surface Temperature".format(city))
        plt.xlabel('Land Surface Temperature ^oC')
        plt.savefig('fig/working/density/lst_{}.pdf'.format(city), format='pdf', dpi=1000, transparent=True)
        plt.clf()

    # scatter plot thermal radiance against land surface, colored by city
    # bmap = brewer2mpl.get_map('Paired','Qualitative',4).mpl_colors
    with plt.style.context('fivethirtyeight'):
        # f = plt.plot()
        for i in range(len(cities)):
            city = cities[i]
            df_city = df.loc[df['city']==city]
            plt.scatter(df_city['tr_day_mean'], df_city['lst_day_mean_mean'], label = city)#, c = bmap[i])
        plt.legend(loc='lower right')
        plt.title('Diurnal')
        plt.xlabel('Thermal Radiance')
        plt.ylabel('Land Surface Temperature')
        plt.savefig('fig/working/density/lst-vs-tr_day.pdf', format='pdf', dpi=1000, transparent=True)
        plt.clf()

    with plt.style.context('fivethirtyeight'):
        # f = plt.plot()
        for i in range(len(cities)):
            city = cities[i]
            df_city = df.loc[df['city']==city]
            plt.scatter(df_city['tr_nght_mean'], df_city['lst_night_mean_mean'], label = city)#, c = bmap[i])
        plt.legend(loc='lower right')
        plt.title('Nocturnal')
        plt.xlabel('Thermal Radiance')
        plt.ylabel('Land Surface Temperature')
        plt.savefig('fig/working/density/lst-vs-tr_night.pdf', format='pdf', dpi=1000, transparent=True)
        plt.clf()

def plot_holdout_points(loss, grid_size):
    '''
    plot the city holdout validation metrics
    '''
    loss['city'] = loss['hold_num'].str[-3:]
    g = sns.factorplot(y="error", x="time_of_day", hue="city", col = "error_metric", data=loss, sharey = False,
                       row = 'model',
                      linestyles='', markers=['$B$','$D$','$X$','$P$'],
                      hue_order = ['bal', 'det', 'phx', 'por'],
                      ci=None)
    # plt.legend(loc='lower center', ncol=4, frameon=False)
    g.set_titles('{row_name}')
    for i, ax in enumerate(g.axes.flat): # set every-other axis for testing purposes
        if i%2==1:
            ax.set_ylim(0,5)
            ax.set_ylabel('mean absolute error')
            ax.set_xlabel('')
        elif i%2==0:
            ax.set_ylim(-1,1)
            ax.set_ylabel('out-of-bag R$^2$')
            ax.set_xlabel('')
    plt.savefig('fig/working/regression/cities_holdout_{}.pdf'.format(grid_size), format='pdf', dpi=1000, transparent=True)
    plt.show()
    plt.clf()

def plot_holdouts(loss, grid_size):
    '''
    plot boxplots of holdouts
    '''
    loss = loss.sort_values(by='model')
    font_scale = 1.75
    with sns.plotting_context("paper", font_scale=font_scale):
        plt.figure(figsize=(width_2col, height_2c))
        g = sns.catplot(y="error", x="time_of_day", hue="model",
                        col = "error_metric", data=loss, sharey = False,
                        col_order = ['r2','mae'],
                        order = ['night\n(mean)','day\n(mean)','night\n(max)','day\n(max)'],
                        kind="box",
                        height = height_2c, aspect = aspect_2c,
                        legend = False)
        g.set_titles('')
        for i, ax in enumerate(g.axes.flat): # set every-other axis for testing purposes
            if i%2==1:
                ax.set_ylim(0,5)
                ax.set_ylabel('mean absolute error ($^o$C)',size=font_size*1.5)
                ax.set_xlabel('')
            elif i%2==0:
                if grid_size == 100:
                    ax.set_ylim(0.5,1)
                else:
                    ax.set_ylim(0,1)
                ax.set_ylabel('out-of-bag R$^2$',size=font_size*1.5)
                ax.set_xlabel('')
                # plt.gca().invert_yaxis()
        plt.savefig('fig/report/holdout_results_{}.pdf'.format(grid_size), format='pdf', dpi=500, transparent=True)
        plt.show()
        plt.clf()

def plot_importance(results_swing, grid_size):
    '''
    plot the feature importance of the variables and the cities
    '''
    # order features by nocturnal swing
    results_swing = results_swing.replace(feature_names)
    feature_order = list(results_swing[results_swing.dependent=='lst_night_mean'].groupby('independent').mean().sort_values(by=('swing'),ascending=False).index)

    results_swing = results_swing.replace(model_names)

    # plot
    # font_size = 15
    font_scale = 3#1.75
    with sns.plotting_context("paper", font_scale=font_scale):
    # sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":font_size,"axes.labelsize":font_size})
    # plt.figure(figsize=(width_2col, height_2c))
        g = sns.factorplot(x='swing', y='independent', hue='dependent',
                            data=results_swing, kind='bar', col='model',
                            order = feature_order,
                            hue_order=['lst_night_mean','lst_day_mean'],
                            col_order=['random forest','gradient boosted\ntrees',
                                        'multivariate adaptive\nspline (mars)',
                                        'generalized additive\n(gam)',
                                        'multivariate linear'],
                            height = height_2c*2,
                            aspect = 0.75
                            # col_wrap = 3
                            )

        g.set_axis_labels("variable influence", "")
        g.set_titles("{col_name}",size=font_size*2.5)
        # g.tick_params(labelsize=font_size)

        new_labels = ['nocturnal','diurnal']
        for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

        # fig = plt.gcf()
        # fig.set_size_inches(15,20)

        plt.savefig('fig/report/variableImportance_{}.pdf'.format(grid_size), format='pdf', dpi=500, transparent=True)
        plt.show()
        plt.clf()
    return(feature_order)

def plot_importance_max(df, grid_size):
    '''
    plot the feature importance of the variables and the cities
    '''
    df = df.replace(feature_names)
    df = df.replace(model_names)
    # order features by nocturnal swing
    for dep in ['lst_night_mean','lst_day_mean','lst_night_max','lst_day_max']:
        results_swing = df.copy()
        results_swing = results_swing[results_swing.dependent == dep]
        feature_order = list(results_swing[results_swing.dependent==dep].groupby('independent').mean().sort_values(by=('swing'),ascending=False).index)

        # plot
        # font_size = 15
        font_scale = 1.5#1.75
        with sns.plotting_context("paper", font_scale=font_scale):
        # sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":font_size,"axes.labelsize":font_size})
        # plt.figure(figsize=(width_2col, height_2c))
            g = sns.catplot(x='swing', y='independent', hue='dependent',
                                data=results_swing, kind='bar', col='model',
                                order = feature_order,
                                # row_order=['lst_night_mean','lst_day_mean','lst_night_max','lst_day_max'],
                                col_order=['random forest',
                                            'convolutional\nneural network',
                                            'gradient boosted\ntrees',
                                            'multivariate adaptive\nspline (mars)',
                                            'generalized additive\n(gam)',
                                            'multivariate linear'],
                                height = height_2c,
                                aspect = 0.60,
                                # col_wrap = 3
                                legend = False
                                )

            g.set_axis_labels("variable influence", "")
            g.set_titles("{col_name}",size=font_size*2.5/2)
            g.set(xlim=(0, 0.4))
            # g.tick_params(labelsize=font_size)

            # new_labels = ['nocturnal_max','diurnal_max']
            # for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

            # fig = plt.gcf()
            # fig.set_size_inches(15,20)

            plt.savefig('fig/working/variableImportance_{}_{}.pdf'.format(dep, grid_size), format='pdf', dpi=500, transparent=True)
            plt.show()
            plt.clf()
    return(feature_order)

def plot_dependence(importance_order, reg_gbm, cities, X_train, vars_selected, show_plot=False):
    '''
    Plot the partial dependence for the different regressors
    '''
    cities =  cities.copy()
    cities.append('all')
    # plot setup (surely this can be a function)

    # init subplots (left is nocturnal, right is diurnal)
    fig, axes = plt.subplots(6, 2, figsize = (15,30), sharey=True)#'row')
    # loop through the top n variables by nocturnal importance
    feature = 0
    for var_dependent in importance_order:
        left_right = 0
        for period in ['nocturnal', 'diurnal']:
            for city in cities:
                gbm = reg_gbm[period][city]
                # feature position
                feature_num = vars_selected.index(var_dependent)
                # calculate the partial dependence
                y, x = partial_dependence(gbm, feature_num, X = X_train[city],
                                        grid_resolution = 100)
                # add the line to the plot
                if city=='all':
                    axes[feature, left_right].plot(x[0],y[0],label=city, linestyle='--', color='#8b8b8b')
                else:
                    axes[feature, left_right].plot(x[0],y[0],label=city)
                # add the label to the plot
                axes[feature, left_right].set_xlabel(var_dependent)
            left_right += 1
        feature += 1
    # legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    # l = plt.legend(handles[0:5], labels[0:5], loc='lower left')
    fig.legend(handles[0:5], labels[0:5], loc='lower center', bbox_to_anchor=(0.5,-0.007),
              fancybox=True, shadow=True, ncol=5)
    # save the figure
    fig.tight_layout()
    if show_plot:
        fig.show()
    else:
        fig.savefig('fig/working/partial_dependence.pdf', format='pdf', dpi=1000, transparent=True)
        fig.clf()

def plot_dependence_city(grid_size):
    '''
    Plot the partial dependence for the different regressors
    '''
    # import data
    pdp_results = pd.read_csv('data/regression/city_{}/results_partial_dependence_{}.csv'.format(grid_size, grid_size))
    # undo the normalization
    normalize_parameters = pd.read_csv('data/normalization_parameters_{}.csv'.format(grid_size))
    normalize_parameters = normalize_parameters.set_index('feature')
    for index, row in pdp_results.iterrows():
        feature = row.independent
        pdp_results.loc[index,'x'] = row['x']*normalize_parameters.loc[feature,'sd'] + normalize_parameters.loc[feature,'mean']
    # humanize the names
    pdp_results = pdp_results.replace(feature_names)
    pdp_results = pdp_results.replace(model_names)
    if grid_size == 500:
        var_interests = {'lst_day_mean':['lcov_11', 'ndvi_mean', 'tree_mean', 'alb_mean','dsm_mean'],
                        'lst_night_mean':['tree_mean', 'ndvi_mean', 'lcov_11', 'svf_mean', 'dsm_mean'],
                        'lst_day_max':['tree_min', 'ndvi_mean','alb_mean','lcov_11','nbdi_max'],
                        'lst_night_max':['tree_min', 'ndvi_mean','tree_mean','dsm_mean','svf_mean']}
    else:
        var_interests = {'lst_day_mean':['tree_mean', 'ndvi_mean', 'alb_mean','dsm_mean','svf_mean'],
                        'lst_night_mean':['tree_mean', 'ndvi_mean', 'alb_mean', 'svf_mean', 'dsm_mean'],
                        'lst_day_max':['ndvi_mean','tree_mean','alb_mean','tree_sd','dsm_mean'],
                        'lst_night_max':['tree_mean', 'ndvi_mean','tree_sd','svf_mean','alb_mean']}
    # plot
    for dep in ['lst_night_mean','lst_day_mean','lst_night_max','lst_day_max']:
        feature_order = var_interests[dep]
        feature_order = [feature_names[x] for x in feature_order]
        pdp_results_plot = pdp_results[pdp_results.dependent == dep]
        # plot
        # font_size = 15
        font_scale = 1.5#1.75
        with sns.plotting_context("paper", font_scale=font_scale):
        # sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":font_size,"axes.labelsize":font_size})
        # plt.figure(figsize=(width_2col, height_2c))
            g = sns.FacetGrid(pdp_results_plot, col="independent", hue = 'city',
                                sharex=False, col_order = feature_order,
                                height = height_2c,
                                aspect = 1
                                )
            g = g.map(plt.plot, 'x', 'mean')
            g.set_titles('')
            axes = g.axes[0]
            i = 0
            for ax in axes:
                ax.axhline(0, ls='--', color='red')
                ax.set_xlabel(feature_order[i])
                i += 1

            plt.savefig('fig/working/pdp_city_{}_{}.pdf'.format(dep, grid_size), format='pdf', dpi=500, transparent=True)
            plt.show()
            plt.clf()

def scatter_lst(df, cities, grid_size):
    '''
    scatter lst night vs day
    '''
    df = df.replace(city_names)
    cities = [city_names[i] for i in cities]
    # scatter plot thermal radiance against land surface, colored by city
    # bmap = brewer2mpl.get_map('Paired','Qualitative',4).mpl_colors
    # with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(width_2col, height_2c))
    for i in range(len(cities)):
        city = cities[i]
        df_city = df.loc[df['city']==city]
        plt.scatter(df_city['lst_day_mean'], df_city['lst_night_mean'], label = city, alpha = 0.5)
        print(df_city['lst_day_mean'].corr(df_city['lst_night_mean']))
    plt.legend(loc='lower right')
    plt.xlabel('$\Delta$mean, average daytime LST ($^o$C)')
    plt.ylabel('$\Delta$mean, average nighttime LST ($^o$C)')
    plt.xlim(-25,15)
    plt.ylim(-15,15)
    plt.text(-23, 10,'correlation = {0:.2f}'.format(df['lst_day_mean'].corr(df['lst_night_mean'])), ha='left', va='top')
    plt.savefig('fig/report/lst_night-vs-day_{}.png'.format(grid_size), format='png', dpi=300, transparent=True)
    plt.show()
    plt.clf()


def scatter_maxlst(df, cities, grid_size):
    '''
    scatter max lst night vs day
    '''
    df = df.replace(city_names)
    cities = [city_names[i] for i in cities]
    # scatter plot thermal radiance against land surface, colored by city
    # bmap = brewer2mpl.get_map('Paired','Qualitative',4).mpl_colors
    # with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(width_2col, height_2c))
    for i in range(len(cities)):
        city = cities[i]
        df_city = df.loc[df['city']==city]
        plt.scatter(df_city['lst_day_max'], df_city['lst_night_max'], label = city, alpha = 0.5)
        print(df_city['lst_day_max'].corr(df_city['lst_night_max']))
    plt.legend(loc='lower right')
    plt.xlabel('$\Delta$mean, maximum daytime LST ($^o$C)')
    plt.ylabel('$\Delta$mean, maximum nighttime LST ($^o$C)')
    plt.xlim(-25,15)
    plt.ylim(-15,15)
    plt.text(-23, 10,'correlation = {0:.2f}'.format(df['lst_day_max'].corr(df['lst_night_max'])), ha='left', va='top')
    plt.savefig('fig/report/lst_max_night-vs-day_{}.png'.format(grid_size), format='png', dpi=300, transparent=True)
    plt.show()
    plt.clf()


def joyplot_lst(df, grid_size):
    '''
    joy (ridge) plot of lst night vs day
    '''
    df1 = df[['lst_night_mean','lst_day_mean','city']]
    df1 = df1.replace(city_names)
    df1 = df1.rename(index=str, columns={"lst_night_mean": "nocturnal", "lst_day_mean": "diurnal"})
    df1 = df1.replace([np.inf, -np.inf], np.nan)
    df1 = df1.dropna(axis=0, how='any')
    # with plt.style.context('fivethirtyeight'):
    fig, axes = joypy.joyplot(df1, by='city', ylim='own',legend=True,
                            figsize=(width_2col, height_2c))
    plt.xlabel('land surface temperature ($^{o}$C)')
    plt.savefig('fig/report/joyplot_lst_{}.pdf'.format(grid_size), format='pdf', dpi=300, transparent=True)
    plt.show()
    plt.clf()

def plot_actualVpredict(y, predict_day, predict_night, model, city, target):
    '''
    plot a scatter of predicted vs actual points
    '''
    xy_line = (np.min([y['night_test'],y['day_test']]),np.max([y['night_test'],y['day_test']]))
    # with plt.style.context('fivethirtyeight'):
    plt.scatter(y['day_test'], predict_day, label = 'Diurnal')
    plt.scatter(y['night_test'], predict_night, label = 'Nocturnal')
    plt.plot(xy_line,xy_line, 'k--')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.legend(loc='lower right')
    plt.title('Gradient Boosted Trees \n {}'.format(city))
    plt.savefig('fig/working/regression/actualVpredict_{}_{}_{}.pdf'.format(target, model, city), format='pdf', dpi=1000, transparent=True)
    plt.clf()

def plot_2d_partialdependence(regressor, time_of_day, grid_size, df_x):
    '''
    Plot the 2d partial dependence
    '''
    # two way partial dependence
    figs, axes = plt.subplots(3, 2, figsize = (width_2col, height_2c*2), sharey=False, sharex=False)
    # loop through the top n variables by nocturnal importance
    df_vars = list(df_x)
    if grid_size == 500:
        # different due to variables included
        features = [(1,4),(1,5),(0,4),(0,5),(2,4),(2,5)]
    else:
        features = [(0,6),(0,8),(1,6),(1,8),(2,6),(2,8)]
    # import data to unnormalize
    normalize_parameters = pd.read_csv('data/normalization_parameters_{}.csv'.format(grid_size))
    normalize_parameters = normalize_parameters.set_index('feature')

    for i in range(len(features)):
        plot_row = floor(i/2)
        left_right = np.mod(i,2)
        # calculate the partial dependence
        pdp, ax = partial_dependence(regressor, features[i], X = df_x,
                            grid_resolution = 50)
        # get data
        XX, YY = np.meshgrid(ax[0], ax[1])
        Z = pdp[0].reshape(list(map(np.size, ax))).T
        # unnormalize
        feature_x = df_vars[features[i][0]]
        feature_y = df_vars[features[i][1]]
        XX = XX*normalize_parameters.loc[feature_x,'sd'] + normalize_parameters.loc[feature_x,'mean']
        YY = YY*normalize_parameters.loc[feature_y,'sd'] + normalize_parameters.loc[feature_y,'mean']

        # add the line to the plot
        CS = axes[plot_row, left_right].contour(XX,YY,Z)#, levels=np.arange(-8,4,1))
        axes[plot_row, left_right].clabel(CS, inline=True, fontsize=12)
        # title and axis labels
        axes[plot_row, left_right].set_xlabel(feature_names[feature_x])
        axes[plot_row, left_right].set_ylabel(feature_names[feature_y])
    # save
    plt.savefig('fig/report/pdp_2d_{}_{}.pdf'.format(time_of_day,grid_size), format='pdf', dpi=500, transparent=True)
    plt.show()
    plt.clf()

def scatter_tree_imp(df, cities, grid_size):
    '''
    scatter lst night vs day
    '''
    df = df.replace(city_names)
    cities = [city_names[i] for i in cities]
    # scatter plot thermal radiance against land surface, colored by city
    # bmap = brewer2mpl.get_map('Paired','Qualitative',4).mpl_colors
    # with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(width_2col, height_2c))
    for i in range(len(cities)):
        city = cities[i]
        df_city = df.loc[df['city']==city]
        plt.scatter(df_city['imp_mean'], df_city['tree_mean'], label = city, alpha = 0.5)
        print(df_city['imp_mean'].corr(df_city['tree_mean']))
    plt.legend(loc='lower right')
    plt.xlabel('% impervious surface')
    plt.ylabel('% tree canopy cover')
    plt.text(20, 80,'correlation = {0:.2f}'.format(df['imp_mean'].corr(df['tree_mean'])), ha='left', va='top')
    plt.savefig('fig/report/imp_v_tree_{}.png'.format(grid_size), format='png', dpi=300, transparent=True)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    # profile() # initialise the board
    main()
