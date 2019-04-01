'''
Here are a collection of functions for analyzing the land surface temperature and
thermal radiance from LandSat images of cities
'''

# import libraries
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import code
pd.options.mode.chained_assignment = 'raise'

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

def import_data(grid_size):
    df = pd.read_csv('data/data_regressions_{}_20190324.csv'.format(grid_size))
    df = df.drop('Unnamed: 0', axis=1)
    return(df)


def regressions(df, cities, sim_num):
    '''
    to compare regression out-of-bag accuracy I need to split into test and train
    I also want to scale some of the variables
    '''

    # prep y
    loss = pd.DataFrame()
    # for city in cities:
    df_city = df#[df['city']==city] #df_city = df.loc[df['city']==city]
    predict_quant = 'lst'
    df_city, response = prepare_lst_prediction(df_city)
    # conduct the holdout
    for i in range(sim_num):
        city = str(i)
        # divide into test and training sets
        X_train, X_test, y_train, y_test = split_holdout(df_city, response, test_size=0.20)#, random_state=RANDOM_SEED)
        # drop unnecessary variables
        X_train, X_test = subset_regression_data(X_train, X_test)
        # response values
        y = define_response_lst(y_train, y_test)
        # apply the null model
        loss = regression_null(y, city, predict_quant, loss)
        # now the GradientBoostingRegressor
        loss = regression_gradientboost(X_train, y, X_test, city, predict_quant, loss)
        # finally, multiple linear regression
        loss = regression_linear(X_train, y, X_test, city, predict_quant, loss)
        # join the loss functions
        loss_city = pd.concat([loss_null, loss_mlr, loss_gbm])
        # print(loss_city)
        loss = loss.append(loss_city)
    # save results
    loss.to_csv('data/regression/holdout_results.csv')


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

        city = 'hold-{}'.format(city)
        # apply the null model
        loss_null = regression_null(y, city, predict_quant)
        # now the GradientBoostingRegressor
        loss_gbm = regression_gradientboost(X_train, y, X_test, city, predict_quant)
        # finally, multiple linear regression
        loss_mlr = regression_linear(X_train, y, X_test, city, predict_quant)
        # join the loss functions
        loss_city = pd.concat([loss_null, loss_mlr, loss_gbm])
        # print(loss_city)
        loss = loss.append(loss_city)
    return(loss)


def prepare_lst_prediction(df):
    '''
    to predict for thermal radiance, let's remove land surface temp and superfluous
    thermal radiance values
    '''
    # drop lst
    lst_vars = ['lst_day_mean','lst_night_mean']
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
    # test
    y['day_test'] = y_test['lst_day_mean']
    y['night_test'] = y_test['lst_night_mean']
    return(y)

###
# Regression code
###

def regression_null(y, city, predict_quant, loss):
    '''
    fit the null model for comparison
    '''
    # train the model
    model = 'null'

    # predict the model
    predict_day = np.ones(len(y['day_test'])) * np.mean(y['day_train'])
    predict_night = np.ones(len(y['night_test'])) * np.mean(y['night_train'])

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'null', city, predict_quant)

    # calculate the MAE
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    # code.interact(local=locals())
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)

    # record results
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

    return(loss)


def regression_gradientboost(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'gbrf'
    # train the model
    gbm_day_reg = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
    gbm_night_reg = GradientBoostingRegressor(max_depth=2, learning_rate=0.1, n_estimators=500, loss='ls')
    # code.interact(local = locals())
    gbm_day_reg.fit(X_train, y['day_train'])
    gbm_night_reg.fit(X_train, y['night_train'])

    # predict the model
    predict_day = gbm_day_reg.predict(X_test)
    predict_night = gbm_night_reg.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)

    # record results
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

    return(loss)


def regression_linear(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the multiple linear regressions
    '''
    model = 'mlr'
    # train the model
    mlr_day_reg = LinearRegression()
    mlr_night_reg = LinearRegression()
    mlr_day_reg.fit(X_train, y['day_train'])
    mlr_night_reg.fit(X_train, y['night_train'])


    # predict the model
    predict_day = mlr_day_reg.predict(X_test)
    predict_night = mlr_night_reg.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'mlr', city, predict_quant)

    # calculate the MAE
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)

    # record results
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

    return(loss)


def regression_randomforest(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'rf'
    # train the model
    reg_day = RandomForestRegressor(n_estimators=500, max_features=1/3)
    reg_night = RandomForestRegressor(n_estimators=500, max_features=1/3)
    reg_day.fit(X_train, y['day_train'])
    reg_night.fit(X_train, y['night_train'])

    # predict the model
    predict_day = reg_day.predict(X_test)
    predict_night = reg_night.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)

    # record results
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

    return(loss)


def regression_mars(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'mars'
    # train the model
    reg_day = Earth(max_degree=1, penalty=1.0, endspan=5)
    reg_night = Earth(max_degree=1, penalty=1.0, endspan=5)
    reg_day.fit(X_train, y['day_train'])
    reg_night.fit(X_train, y['night_train'])

    # predict the model
    predict_day = reg_day.predict(X_test)
    predict_night = reg_night.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)

    # record results
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

    return(loss)


def regression_gam(X_train, y, X_test, city, predict_quant, loss):
    '''
    fit the GradientBoostingRegressor
    '''
    model = 'gam'
    # train the model
    reg_day = LinearGAM(n_splines=10)
    reg_night = LinearGAM(n_splines=10)
    reg_day.fit(X_train, y['day_train'])
    reg_night.fit(X_train, y['night_train'])

    # predict the model
    predict_day = reg_day.predict(X_test)
    predict_night = reg_night.predict(X_test)

    # plot predict vs actual
    plot_actualVpredict(y, predict_day, predict_night, 'gbrf', city, predict_quant)

    # calculate the error metrics
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)

    # record results
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

def plot_holdout_points(loss):
    loss_plot = loss.unstack(level=0)
    loss_plot = loss_plot.unstack(level=0)
    loss_plot = pd.DataFrame(loss_plot).T
    loss_plot = pd.melt(loss_plot)
    loss_plot['city'] = loss_plot['city'].str[-3:]
    # print(loss_plot)
    # with plt.style.context('fivethirtyeight'):
    five_thirty_eight = [
        "#30a2da",
        "#fc4f30",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",
    ]
    sns.set_palette(five_thirty_eight)
    mpl.rcParams.update({'font.size': 20})
    g = sns.factorplot(orient="h", y="model", x="value", hue="city", linestyles='', markers=['$B$','$D$','$X$','$P$'],
    col = "loss", row = "time", data=loss_plot, sharex = 'col', order=['null','mlr','gbm'], aspect=2, scale=2.5)
    g.set_titles('{row_name}')
    for i, ax in enumerate(g.axes.flat): # set every-other axis for testing purposes
            if i==2:
                ax.set_xlim(0,12)
                ax.set_xlabel('Mean Absolute Error')
            elif i==3:
                ax.set_xlim(-1,1)
                ax.set_xlabel('Out-of-bag R$^2$')
    plt.savefig('fig/working/regression/cities_holdout.pdf', format='pdf', dpi=1000, transparent=True)
    plt.show()
    plt.clf()

def plot_holdouts(sim_num):
    '''
    plot boxplots of holdouts
    '''
    loss = pd.read_csv('data/regression/holdout_results.csv', index_col=[0,1],header=[0,1], skipinitialspace=True,keep_default_na=False )
    loss['hold_id'] = loss.city #np.tile(np.repeat(range(sim_num),3),4)
    loss.set_index('hold_id',append=True,inplace=True)
    loss.reorder_levels(['hold_id', 'model'])#, 'city'])
    for error_type in ['r2', 'mae']:
        loss_type = loss.copy()
        loss_type = loss_type.xs(error_type, axis=1, level=1)
        loss_type = loss_type.unstack(level=0)
        loss_type = loss_type.unstack(level=0)
        loss_type = pd.DataFrame(loss_type).T
        loss_type = pd.melt(loss_type)
        loss_type[error_type] = loss_type['value']
        # print(loss_type)
        five_thirty_eight = [
            "#30a2da",
            "#fc4f30",
            "#e5ae38",
            "#6d904f",
            "#8b8b8b",
        ]
        sns.set_palette(five_thirty_eight)
        mpl.rcParams.update({'font.size': 22})
        g = sns.factorplot(orient="h", x=error_type, y='model', col="time", data=loss_type, kind="box", order=['null','mlr','gbm'], aspect=2, palette = five_thirty_eight)
        for i, ax in enumerate(g.axes.flat): # set every-other axis for testing purposes
            if i>5:
                if error_type=='r2':
                    ax.set_xlabel('Out-of-bag R$^2$')
                else:
                    ax.set_xlabel('Mean Absolute Error')
        for i, ax in enumerate(g.axes.flat):
            i = 0
            for patch in ax.artists:
                patch.set_facecolor(five_thirty_eight[i])
                i += 1
        plt.savefig('fig/working/regression/holdout_results_{}.pdf'.format(error_type), format='pdf', dpi=1000, transparent=True)
        plt.show()
        plt.clf()

def plot_importance(reg_gbm, cities, show_plot=False):
    '''
    plot the feature importance of the variables and the cities
    '''
    cities =  cities.copy()
    cities.append('all')
    five_thirty_eight = [
        "#30a2da",
        "#fc4f30",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",]
    sns.set_palette(five_thirty_eight)
    mpl.rcParams.update({'font.size': 20})
    # get the covariates - these will be the indices in the dataframe
    header = pd.MultiIndex.from_product([['diurnal','nocturnal'], cities],
                                    names=['time','city'])
    var_imp = pd.DataFrame(columns = header, index = list(reg_gbm['covariates']))
    # loop the cities to add the var imp to the df
    for city in cities:
        for time in ['diurnal','nocturnal']:
            var_imp.loc[:,(time, city)] = reg_gbm[time][city].feature_importances_
    var_imp.loc[:,('nocturnal','mean')] = np.mean(var_imp.loc[:,'nocturnal'].drop('all',axis=1),axis=1)
    var_imp = var_imp.sort_values(by=('nocturnal','mean'),ascending=False)
    # make the nocturnal values negative
    nocturnal = var_imp.loc[:,'nocturnal'].copy()
    nocturnal = nocturnal.drop('mean',axis=1)
    nocturnal['covariate'] = nocturnal.index
    nocturnal = pd.melt(nocturnal,id_vars=['covariate'])
    nocturnal.loc[:,'value'] = nocturnal['value'] * -1
    # diurnal
    diurnal = var_imp.loc[:,'diurnal'].copy()
    diurnal.loc[:,'covariate'] = diurnal.index
    diurnal = pd.melt(diurnal,id_vars=['covariate'])
    # plot
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 9)
    ax = sns.barplot(orient="h", y='covariate', x='value',hue='city', data=nocturnal, palette = five_thirty_eight)
    ax = sns.barplot(orient="h", y='covariate', x='value',hue='city', data=diurnal, palette = five_thirty_eight)
    plt.xlabel('Variable Importance')
    plt.ylabel('Variables')
    # legend
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:len(cities)], labels[0:len(cities)], loc='lower right')
    # zero line
    [plt.axvline(_x, linewidth=0.5, color='k', linestyle='--') for _x in np.arange(-0.3, 0.4, 0.1)]
    plt.axvline(x=0, color='k', linestyle='-', linewidth = 2)
    plt.xlim(-0.4,0.4)
    plt.tight_layout()
    # save the figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('fig/working/variable_importance_selected.pdf', format='pdf', dpi=1000, transparent=True)
        plt.clf()
    importance_order = var_imp.index.values
    return(importance_order)

def plot_dependence(importance_order, reg_gbm, cities, X_train, vars_selected, show_plot=False):
    '''
    Plot the partial dependence for the different regressors
    '''
    cities =  cities.copy()
    cities.append('all')
    # plot setup (surely this can be a function)
    five_thirty_eight = [
        "#30a2da",
        "#fc4f30",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",]
    sns.set_palette(five_thirty_eight)
    mpl.rcParams.update({'font.size': 20})
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

def scatter_lst(df, cities):
    '''
    scatter lst night vs day
    '''

    # scatter plot thermal radiance against land surface, colored by city
    # bmap = brewer2mpl.get_map('Paired','Qualitative',4).mpl_colors
    with plt.style.context('fivethirtyeight'):
        for i in range(len(cities)):
            city = cities[i]
            df_city = df.loc[df['city']==city]
            plt.scatter(df_city['lst_day_mean_mean'], df_city['lst_night_mean_mean'], label = city, alpha = 0.5)
        plt.legend(loc='lower right')
        plt.xlabel('Day LST ($^o$C)')
        plt.ylabel('Night LST ($^o$C)')
        plt.text(20, 40,'Correlation = {0:.2f}'.format(df_city['lst_day_mean_mean'].corr(df_city['lst_night_mean_mean'])), ha='left', va='top')
        plt.savefig('fig/working/density/lst_night-vs-day.pdf', format='pdf', dpi=300, transparent=True)
        plt.clf()

def joyplot_lst(df, cities):
    '''
    scatter lst night vs day
    '''

    # scatter plot thermal radiance against land surface, colored by city
    # bmap = brewer2mpl.get_map('Paired','Qualitative',4).mpl_colors
    import joypy
    df1 = df[['lst_night_mean_mean','lst_day_mean_mean','city']]
    df1 = df1.rename(index=str, columns={"lst_night_mean_mean": "night", "lst_day_mean_mean": "day"})
    df1 = df1.replace([np.inf, -np.inf], np.nan)
    df1 = df1.dropna(axis=0, how='any')
    with plt.style.context('fivethirtyeight'):
        fig, axes = joypy.joyplot(df1, by='city', ylim='own',legend=True)
        plt.xlabel('Land Surface Temperature ($^o$C)')
    plt.savefig('fig/working/density/joyplot_lst.pdf', format='pdf', dpi=300, transparent=True)
    plt.clf()

def plot_actualVpredict(y, predict_day, predict_night, model, city, target):
    '''
    plot a scatter of predicted vs actual points
    '''
    xy_line = (np.min([y['night_test'],y['day_test']]),np.max([y['night_test'],y['day_test']]))
    with plt.style.context('fivethirtyeight'):
        plt.scatter(y['day_test'], predict_day, label = 'Diurnal')
        plt.scatter(y['night_test'], predict_night, label = 'Nocturnal')
        plt.plot(xy_line,xy_line, 'k--')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend(loc='lower right')
        plt.title('Gradient Boosted Trees \n {}'.format(city))
        plt.savefig('fig/working/regression/actualVpredict_{}_{}_{}.pdf'.format(target, model, city), format='pdf', dpi=1000, transparent=True)
        plt.clf()


if __name__ == '__main__':
    # profile() # initialise the board
    main()
