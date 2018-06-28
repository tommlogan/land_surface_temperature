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
pd.options.mode.chained_assignment = 'raise'

# regression libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor

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
    df = import_data(cities)

    # data transformations
    df = transform_data(df)

    # present the data - density plot
    plot_density(df, cities)

    # regression
    ## train on three cities, test on one
    loss = regression_cityholdouts(df, cities)
    # plot the points
    plot_holdout_points(loss)

    ## for each city, train and test
    sim_num = 100 # number of holdouts
    regressions(df, cities, sim_num)
    plot_holdouts()

    # variable importance and partial dependence
    reg_gbm = full_gbm_regression(df, cities)

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

def import_data(cities):
    df = pd.DataFrame()
    for city in cities:
        # import city data
        df_new = pd.read_csv('data/processed/grid/{}/2018-04-14/{}_data.csv'.format(city,city))
        # append city name to df
        df_new['city'] = city
        # bind to complete df
        df = df.append(df_new, ignore_index=True)
    return(df)

def transform_data(df):
    '''
    apply a couple of transformations to the data
    '''

    # Remove the lat, long, name, cId columns and area == 0 rows
    df = df.drop(['Unnamed: 0', 'x', 'y', 'cId'], axis = 1)

    ##
    # Land cover
    ##

    # drop lcov 0 column
    df = df.drop(['lcov_0','lcov_0_sl'], axis=1)

    # list of the variables
    vars_all = df.columns.values
    # which columns have an lcov values in them?
    vars_lcov = [i for i in vars_all if 'lcov' in i]

    # set other nan landcover values to 0
    df[vars_lcov] = df[vars_lcov].fillna(0)

    # make the land cover variables a percentage , rather than the total m2.
    # calculate the max area of a cell
    area_max = np.max([np.max(df[i]) for i in vars_lcov])
    # divide these values by the area value
    df = df[df['area'] > 0]
    for var_lcov in vars_lcov:
        df[var_lcov] = df[var_lcov]/area_max

    # drop rows with water more than 20% of area
    # df = df.loc[df['lcov_11'] < 0.2]

    # Drop the 2013 thermal radiance measure (this is in bal dataset for validation)
    tr_2013 = [s for s in df.columns.values if 'tr_2013' in s]
    df = df.drop(tr_2013, axis=1)

    # drop columns which are all nan
    df = df.dropna(axis=1, how='all')
    # drop any nan rows and report how many dropped
    df = df.dropna(axis=0, how='any')

    # scale albedo
    vars_alb = [i for i in vars_all if 'alb' in i]
    alb_min = df[vars_alb].values.min()
    alb_max = df[vars_alb].values.max()
    df.loc[:,vars_alb] = (df[vars_alb]-alb_min)/(alb_max-alb_min)

    # make tree values a percentage
    vars_tree = [i for i in vars_all if 'tree' in i]
    df.loc[:,vars_tree] = df[vars_tree]/100

    ###
    # elevation transform and scale
    ###
    vars_elev = [i for i in vars_all if 'elev' in i]
    cities = np.unique(df['city'])
    # subtract city median elevation
    medians = df.groupby(df.city)[vars_elev].median().median(axis=1)
    for city in cities:
        df.loc[df['city']==city,vars_elev] = df.loc[df['city']==city,vars_elev] - medians[city]
    # scale the cities
    elev_max = np.max(df[vars_elev].values.max())
    elev_min = np.min(df[vars_elev].values.min())
    df.loc[:,vars_elev] = (df.loc[:,vars_elev] - elev_min)/(elev_max-elev_min)

    return(df)

def regressions(df, cities, sim_num):
    '''
    to compare regression out-of-bag accuracy I need to split into test and train
    I also want to scale some of the variables
    '''

    # prep y
    loss = pd.DataFrame()
    for city in cities:
        df_city = df[df['city']==city] #df_city = df.loc[df['city']==city]
        predict_quant = 'lst'
        df_city, response = prepare_lst_prediction(df_city)
        # conduct the holdout
        for i in range(sim_num):
            # divide into test and training sets
            X_train, X_test, y_train, y_test = train_test_split(df_city, response, test_size=0.2)#, random_state=RANDOM_SEED)
            # scale explanatory variables
            X_train, X_test = scale_X(X_train, X_test)
            # response values
            y = define_response_lst(y_train, y_test)
            # import code
            # code.interact(local=locals())
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
    loss.to_csv('data/regression/holdout_results.csv')

def regression_cityholdouts(df, cities):
    '''
    to compare regression out-of-bag accuracy I need to split into test and train
    I also want to scale some of the variables
    '''
    predict_quant = 'lst'
    # normalize y
    df = normalize_response(cities, df, predict_quant)
    # prep y
    df, response = prepare_lst_prediction(df)
    loss = pd.DataFrame()
    for city in cities:
        train_idx = np.where(df['city'] != city)
        test_idx = np.where(df['city'] == city)
        # import code
        # code.interact(local=locals())
        # divide into test and training sets
        X_train = df.iloc[train_idx].copy()
        y_train = response.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()
        y_test = response.iloc[test_idx].copy()
        # scale explanatory variables
        X_train, X_test = scale_X(X_train, X_test)
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

def prepare_tr_prediction(df):
    '''
    to predict for thermal radiance, let's remove land surface temp and superfluous
    thermal radiance values
    '''
    lst_mean = df[['lst_day_mean_mean','lst_night_mean_mean', 'lst_day_mean_min', 'lst_day_mean_max', 'lst_night_mean_min', 'lst_night_mean_max']]
    lst_vars = ['lst_day_mean_mean','lst_night_mean_mean', 'lst_day_mean_min', 'lst_day_mean_max', 'lst_night_mean_min', 'lst_night_mean_max']
    lst_vars_sl = [var + '_sl' for var in lst_vars]
    df = df.drop(lst_vars, axis=1)
    df = df.drop(lst_vars_sl, axis=1)

    # drop additional thermal radiance variables
    thermrad_vars = ['tr_day_mean','tr_nght_mean', 'tr_day_min', 'tr_nght_min', 'tr_day_max', 'tr_nght_max']
    thermrad_mean = df[thermrad_vars]

    thermrad_vars_sl = [var + '_sl' for var in thermrad_vars]
    df = df.drop(thermrad_vars, axis=1)
    df = df.drop(thermrad_vars_sl, axis=1)

    return(df, thermrad_mean)

def prepare_lst_prediction(df):
    '''
    to predict for thermal radiance, let's remove land surface temp and superfluous
    thermal radiance values
    '''
    # drop lst
    lst_vars = ['lst_day_mean_mean','lst_night_mean_mean', 'lst_day_mean_min', 'lst_day_mean_max', 'lst_night_mean_min', 'lst_night_mean_max']
    lst_mean = df[lst_vars]
    lst_vars_sl = [var + '_sl' for var in lst_vars]
    df = df.drop(lst_vars, axis=1)
    df = df.drop(lst_vars_sl, axis=1)

    # drop additional thermal radiance variables
    thermrad_vars = ['tr_day_mean','tr_nght_mean', 'tr_day_min', 'tr_nght_min', 'tr_day_max', 'tr_nght_max']
    thermrad_mean = df[thermrad_vars]
    thermrad_vars_sl = [var + '_sl' for var in thermrad_vars]
    df = df.drop(thermrad_vars, axis=1)
    df = df.drop(thermrad_vars_sl, axis=1)

    # drop lcov variables
    vars_lcov = [i for i in df.columns.values if 'lcov' in i]
    # keep water (lcov_11)
    vars_lcov = [i for i in vars_lcov if '11' not in i]
    df = df.drop(vars_lcov, axis=1)

    # drop impervious variables because they are 1:1 correlated with tree canopy
    vars_imp = [i for i in df.columns.values if 'imp' in i]
    df = df.drop(vars_imp, axis=1)

    return(df, lst_mean)

def scale_X(X_train, X_test):
    '''
    scale the variables so they are more suited for regression
    '''
    vars_all = X_train.columns.values
    cities = np.unique(X_train['city'])

    # scaler = preprocessing.MinMaxScaler()
    # scaler.fit(X_train)
    # X_scaled = scaler.transform(X_train)
    # X_train = pd.DataFrame(data = X_scaled, columns = X_train.columns.values)
    # X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns.values)
    # vars_elev = [i for i in vars_all if 'elev' in i]
    # if len(vars_elev)>0:
    #     # print(cities)
    #     # print(len(cities))
    #     if len(cities) > 1:
    #         # normalize elevation by subtracting median of city from the city
    #         df = pd.concat([X_train, X_test])
    #         medians = df.groupby(df.city)[vars_elev].median().median(axis=1)
    #         for city in cities:
    #             X_train.loc[X_train['city']==city,vars_elev] = X_train.loc[X_train['city']==city,vars_elev] - medians[city]
    #             X_test.loc[X_test['city']==city,vars_elev] = X_test.loc[X_test['city']==city,vars_elev] - medians[city]
    #
    #     X_train = X_train.drop('city', axis=1)
    #     X_test = X_test.drop('city', axis=1)
    #
    #     if len(X_test[vars_elev].values) > 0:
    #         elev_max = np.max([X_train[vars_elev].values.max(), X_test[vars_elev].values.max()])
    #         elev_min = np.min([X_train[vars_elev].values.min(), X_test[vars_elev].values.min()])
    #         X_test.loc[:,vars_elev] = (X_test.loc[:,vars_elev] - elev_min)/(elev_max-elev_min)
    #     else:
    #         elev_max = X_train[vars_elev].values.max()
    #         elev_min = X_train[vars_elev].values.min()
    #     # print(elev_max, elev_min)
    #     X_train.loc[:,vars_elev] = (X_train.loc[:,vars_elev] - elev_min)/(elev_max-elev_min)
    # else:
    X_train = X_train.drop('city', axis=1)
    X_test = X_test.drop('city', axis=1)

    return(X_train, X_test)

def define_response_tr(y_train, y_test):
    y = {}
    y['day_train'] = y_train['tr_day_mean']
    y['night_train'] = y_train['tr_nght_mean']
    # test
    y['day_test'] = y_test['tr_day_mean']
    y['night_test'] = y_test['tr_nght_mean']
    return(y)

def define_response_lst(y_train, y_test):
    y = {}
    y['day_train'] = y_train['lst_day_mean_mean']
    y['night_train'] = y_train['lst_night_mean_mean']
    # test
    y['day_test'] = y_test['lst_day_mean_mean']
    y['night_test'] = y_test['lst_night_mean_mean']
    return(y)

def normalize_response(cities, df, predict_quant):
    '''
    normalize the response data so it can be predicted between cities
    '''
    # identify keys
    if predict_quant=='lst':
        vars = ['lst_day_mean_mean','lst_night_mean_mean', 'lst_day_mean_min', 'lst_day_mean_max', 'lst_night_mean_min', 'lst_night_mean_max']
    else:
        vars = ['tr_day_mean','tr_nght_mean', 'tr_day_min', 'tr_nght_min', 'tr_day_max', 'tr_nght_max']

    # loop keys
    for k in vars:
        # loop through the cities
        for city in cities:
            # calculate the mean
            k_mean = np.mean(df.loc[df['city']==city,k])
            # calculate the sd
            k_std = np.std(df.loc[df['city']==city,k])
            # normalize the data
            df.loc[df['city']==city,k] = df.loc[df['city']==city,k] - k_mean
    return(df)

def regression_null(y, city, predict_quant):
    '''
    fit the null model for comparison
    '''
    # train the model

    # predict the model
    predict_day = np.ones(len(y['day_test'])) * np.mean(y['day_train'])
    predict_night = np.ones(len(y['night_test'])) * np.mean(y['night_train'])

    with plt.style.context('fivethirtyeight'):
        # plot predict vs actual
        plt.scatter(y['day_test'], predict_day, label = 'Diurnal')
        plt.scatter(y['night_test'], predict_night, label = 'Nocturnal')
        xy_line = (np.min(y['night_train']),np.max(y['day_train']))
        plt.plot(xy_line,xy_line, 'k--')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend(loc='lower right')
        plt.title('Null model for {}'.format(city))
        plt.savefig('fig/working/regression/actualVpredict_{}_null_{}.pdf'.format(predict_quant, city), format='pdf', dpi=1000, transparent=True)
        plt.clf()

    # calculate the MAE
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)
    # print('\n \nNull model for {}'.format(city))
    # print('Nocturnal \n MAE: {:.4f} \n Out-of-bag R^2: {:.2f}'.format(mae_night, r2_night))
    # print('Diurnal \n MAE: {:.4f} \n Out-of-bag R^2: {:.2f}'.format(mae_day, r2_day))
    rownames = pd.MultiIndex(levels = [['null'], [city]], labels=[[0],[0]], names = ['model','city'])
    header = pd.MultiIndex.from_product([['diurnal','nocturnal'],
                                     ['mae','r2']],
                                    names=['time','loss'])
    loss = pd.DataFrame([[mae_day, r2_day, mae_night, r2_night]], columns = header, index = rownames)
    return(loss)

def regression_gradientboost(X_train, y, X_test, city, predict_quant):
    '''
    fit the GradientBoostingRegressor
    '''
    # print(X_train.columns.values)
    # train the model
    gbm_day_reg = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
    gbm_night_reg = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
    gbm_day_reg.fit(X_train, y['day_train'])
    gbm_night_reg.fit(X_train, y['night_train'])

    # predict the model
    predict_day = gbm_day_reg.predict(X_test)
    predict_night = gbm_night_reg.predict(X_test)

    # plot predict vs actual
    xy_line = (np.min(y['night_train']),np.max(y['day_train']))
    with plt.style.context('fivethirtyeight'):
        plt.scatter(y['day_test'], predict_day, label = 'Diurnal')
        plt.scatter(y['night_test'], predict_night, label = 'Nocturnal')
        plt.plot(xy_line,xy_line, 'k--')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend(loc='lower right')
        plt.title('Gradient Boosted Trees model for {}'.format(city))
        plt.savefig('fig/working/regression/actualVpredict_{}_gbrf_{}.pdf'.format(predict_quant, city), format='pdf', dpi=1000, transparent=True)
        plt.clf()

    # calculate the MAE
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)
    # print('\n \nGradient Boosted Trees model for {}'.format(city))
    # print('Nocturnal \n MAE: {:.4f} \n Out-of-bag R^2: {:.2f}'.format(mae_night, r2_night))
    # print('Diurnal \n MAE: {:.4f} \n Out-of-bag R^2: {:.2f}'.format(mae_day, r2_day))
    rownames = pd.MultiIndex(levels = [['gbm'], [city]], labels=[[0],[0]], names = ['model','city'])
    header = pd.MultiIndex.from_product([['diurnal','nocturnal'],
                                     ['mae','r2']],
                                    names=['time','loss'])
    loss = pd.DataFrame([[mae_day, r2_day, mae_night, r2_night]], columns = header, index = rownames)
    return(loss)

def regression_linear(X_train, y, X_test, city, predict_quant):
    '''
    fit the multiple linear regressions
    '''
    # print(X_train.columns.values)
    # train the model
    mlr_day_reg = LinearRegression()
    mlr_night_reg = LinearRegression()
    mlr_day_reg.fit(X_train, y['day_train'])
    mlr_night_reg.fit(X_train, y['night_train'])


    # predict the model
    predict_day = mlr_day_reg.predict(X_test)
    predict_night = mlr_night_reg.predict(X_test)

    # plot predict vs actual
    xy_line = (np.min(y['night_train']),np.max(y['day_train']))
    with plt.style.context('fivethirtyeight'):
        plt.scatter(y['day_test'], predict_day, label = 'Diurnal')
        plt.scatter(y['night_test'], predict_night, label = 'Nocturnal')
        plt.plot(xy_line,xy_line, 'k--')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend(loc='lower right')
        plt.title('Multiple Linear Regression model for {}'.format(city))
        plt.savefig('fig/working/regression/actualVpredict_{}_mlr_{}.pdf'.format(predict_quant,city), format='pdf', dpi=1000, transparent=True)
        plt.clf()

    # calculate the MAE
    mae_day = np.mean(abs(predict_day - y['day_test']))
    mae_night = np.mean(abs(predict_night - y['night_test']))
    r2_day = r2_score(y['day_test'], predict_day)
    r2_night = r2_score(y['night_test'], predict_night)
    # print('\n \nMultiple Linear Regression model for {}'.format(city))
    # print('Nocturnal \n MAE: {:.4f} \n Out-of-bag R^2: {:.2f}'.format(mae_night, r2_night))
    # print('Diurnal \n MAE: {:.4f} \n Out-of-bag R^2: {:.2f}'.format(mae_day, r2_day))
    rownames = pd.MultiIndex(levels = [['mlr'], [city]], labels=[[0],[0]], names = ['model','city'])
    header = pd.MultiIndex.from_product([['diurnal','nocturnal'],
                                     ['mae','r2']],
                                    names=['time','loss'])
    loss = pd.DataFrame([[mae_day, r2_day, mae_night, r2_night]], columns = header, index = rownames)
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
        X_train[city], X_test, y_train, y_test = train_test_split(df_city, response, test_size=0)#, random_state=RANDOM_SEED)
        print(len(X_train[city]), len(X_test))
        # scale explanatory variables
        X_train[city], X_train[city]  = scale_X(X_train[city], X_train[city])
        # response values
        y = define_response_lst(y_train, y_train)
        # fit the model
        reg_gbm['diurnal'][city] = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
        reg_gbm['diurnal'][city].fit(X_train[city], y['day_train'])
        reg_gbm['nocturnal'][city] = GradientBoostingRegressor(max_depth=2, random_state=RANDOM_SEED, learning_rate=0.1, n_estimators=500, loss='ls')
        reg_gbm['nocturnal'][city].fit(X_train[city], y['night_train'])
    reg_gbm['covariates'] = X_train[city].columns
    return(reg_gbm, X_train)

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
                X_train, X_test, y_train, y_test = train_test_split(df_var, response, test_size=0.2)#, random_state=RANDOM_SEED)
                # scale explanatory variables
                X_train, X_test  = scale_X(X_train.copy(), X_test.copy())
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
    g = sns.factorplot(orient="h", y="model", x="value", hue="city", linestyles='', markers=['v','o','^','x'],
    col = "loss", row = "time", data=loss_plot, sharex = 'col', order=['null','mlr','gbm'], aspect=2, scale=1.5)
    for i, ax in enumerate(g.axes.flat): # set every-other axis for testing purposes
            if i==2:
                ax.set_xlim(0,12)
                ax.set_xlabel('Mean Absolute Error')
            elif i==3:
                ax.set_xlim(-1,1)
                ax.set_xlabel('Out-of-bag R$^2$')
    plt.savefig('fig/working/regression/cities_holdout.pdf', format='pdf', dpi=1000, transparent=True)
    plt.clf()

def plot_holdouts():
    '''
    plot boxplots of holdouts
    '''
    loss = pd.read_csv('data/regression/holdout_results.csv', index_col=[0,1],header=[0,1], skipinitialspace=True,keep_default_na=False )
    loss['hold_id'] = np.tile(np.repeat(range(100),3),4)
    loss.set_index('hold_id',append=True,inplace=True)
    loss.reorder_levels(['hold_id', 'model', 'city'])
    for error_type in ['r2', 'mae']:
        loss_type = loss.copy()
        loss_type = loss_type.xs(error_type, axis=1, level=1)
        loss_type = loss_type.unstack(level=0)
        loss_type = loss_type.unstack(level=0)
        # loss_type = pd.DataFrame(loss_type).T
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
        g = sns.factorplot(orient="h", x=error_type, y='model', row="city",col="time", data=loss_type, kind="box", order=['null','mlr','gbm'], aspect=2, palette = five_thirty_eight)
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
    l = plt.legend(handles[0:5], labels[0:5], loc='lower right')
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

if __name__ == '__main__':
    # profile() # initialise the board
    main()
