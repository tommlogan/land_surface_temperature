'''
Integrate the data sets, and scale the variables.
Output is a csv with data for analysis
'''

# import libraries
import pandas as pd
import numpy as np
import time
import code
scale = True
def main(scale=True):

    cities = ['bal','det','phx','por']
    grid_size = 500
    # init all city dataframe
    df = pd.DataFrame()
    add_index = 0
    for city in cities:
        # import city data
        # if grid_size == 100:
        #     dir_date = '2018-10-20'
        # else:
        dir_date = '2020-03-01'
        df_city = pd.read_csv('data/processed/grid/{}/{}/{}_data_{}.csv'.format(city, dir_date, city,grid_size))
        # scale city specific variables
        df_city = scaling_city(df_city, scale)
        # add the grid group numbers
        df_city, add_index = holdout_grid(df_city, add_index)
        # append city name to df
        df_city['city'] = city
        # bind to complete df
        df = df.append(df_city, ignore_index=True)
    # apply transformation on entire dataset
    df = adjust_variables(df, scale)
    if scale:
        df = scaling_all(df, grid_size)
    # write to csv
    df = df.loc[:, df.isnull().mean() < .00001]
    # code.interact(local = locals())
    if scale:
        df.to_csv('data/data_regressions_{}_{}.csv'.format(grid_size, time.strftime("%Y%m%d")))
    else:
        df.to_csv('data/data_regressions_{}_{}_unnormalized.csv'.format(grid_size, time.strftime("%Y%m%d")))


def scaling_city(df_city, scale):
    '''
    scale the variables which are city specific (e.g. elevation has the city mean removed before [0,1] scaling)
    '''

    vars_all = df_city.columns.values

    ### ADMIN ###
    df_city = df_city.drop(['Unnamed: 0', 'cId'], axis = 1)
    # outside city limits
    df_city = df_city[df_city['area'] > 0]

    # drop any values where the landsurface temperature is inf or nan
    df_city = df_city[np.isfinite(df_city['lst_day_mean'])]
    df_city = df_city[np.isfinite(df_city['lst_night_mean'])]
    df_city = df_city[np.isfinite(df_city['lst_night_max'])]
    df_city = df_city[np.isfinite(df_city['lst_day_max'])]

    ## for phoenix - remove large rural areas outside of lidar zone
    if df_city.city.iloc[100] == 'phx':
        df_city = df_city[np.isfinite(df_city['svf_mean'])]

    # setting building NaN values to 0
    df_city.bldg.fillna(0, inplace = True)

    # setting NaN population regions to 0
    df_city = df_city[np.isfinite(df_city['pdens_mean'])]

    # drop a couple of columns
    df_city = df_city.loc[:, df_city.isnull().mean() < .05]
    # drop_vars = ['bldg_sl', 'tree_max_sl', 'dsm_max_sl', 'dsm_mean_sl', 'dsm_min_sl', 'dsm_sd_sl']
    # df_city = df_city.drop(drop_vars, axis=1)

    ### NaN variables
    # remove night time lights (not enough variance to get standard deviation)
    vars_all = df_city.columns.values
    vars_ntl = [x for x in vars_all if 'ntl' in x]
    df_city = df_city.drop(vars_ntl, axis=1)
#     # drop any nan rows and report how many dropped
    df_city = df_city.dropna(axis='index', how='any')

    ### Elevation ###
    # subtract the mean from all of the elevation variables (except standard deviation)
    # list elev vars
    vars_elev = [i for i in vars_all if 'elev' in i]
    # remove sd vars
    vars_elev = [x for x in vars_elev if 'sd' not in x]
    # calculate the mean of each
    elev_means = df_city[vars_elev].mean(axis=0)
    df_city[vars_elev] = df_city[vars_elev] - elev_means

    ### Albedo ###
    # multiply by 100 to get percent
    # list alb vars
    vars_alb = [i for i in vars_all if 'alb' in i]
    # scale
    df_city[vars_alb] = df_city[vars_alb]*100



    ### DSM ###
    # subtract the mean from all of the dsm variables (except standard deviation)
    # list dsm vars
    vars_dsm = [i for i in vars_all if 'dsm' in i]
    # remove sd vars
    vars_dsm = [x for x in vars_dsm if 'sd' not in x]
    # calculate the mean of each
    dsm_means = df_city[vars_dsm].mean(axis=0)
    df_city[vars_dsm] = df_city[vars_dsm] - dsm_means

    ### LST ###
    # drop the lst vars except night and day mean
    vars_response = ['lst_day_mean','lst_night_mean','lst_night_max','lst_day_max']
    vars_lst = [i for i in vars_all if 'lst' in i]
    # remove vars to keep
    vars_lst_drop = [x for x in vars_lst if x not in vars_response]
    # drop vars
    df_city = df_city.drop(vars_lst_drop, axis=1)
    # calculate the mean of each
    if scale:
        lst_means = df_city[vars_response].mean(axis=0)
        df_city[vars_response] = df_city[vars_response] - lst_means

    return(df_city)


def adjust_variables(df, scale):
    vars_all = df.columns.values
    ### Land cover ###
    vars_lcov = [i for i in vars_all if 'lcov' in i]
    # set other NaN landcover values to 0
    df[vars_lcov] = df[vars_lcov].fillna(0)
    # calculate the max area of a cell
    area_max = np.max([np.max(df[i]) for i in vars_lcov])
    # divide these values by the area value
    df = df[df['area'] > 0]
    for var_lcov in vars_lcov:
        df[var_lcov] = df[var_lcov]/area_max*100
    # drop rows with water more than 20% of area
    # df = df.loc[df['lcov_11'] < 0.2]
    # drop lcov variables
    vars_lcov = [i for i in df.columns.values if 'lcov' in i]
    # keep water (lcov_11)
    vars_lcov = [i for i in vars_lcov if '11' not in i]
    df = df.drop(vars_lcov, axis=1)

    ### Drop impervious surface
    if scale:
        vars_imp = [i for i in vars_all if 'imp' in i]
        df = df.drop(vars_imp, axis=1)

    ### Building area
    # set the building area as a percent of the cell's area
    df.bldg = df.bldg/df.area*100


    # drop area
    df = df.drop('area', axis=1)
    return(df)


def scaling_all(df, grid_size):
    '''
    scale the variables between the cities
    '''
    # code.interact(local = locals())
    # Transform to [0,1]
    normalize_parameters = pd.DataFrame()
    vars_all = df.columns.values
    vars_indep = [i for i in vars_all if 'lst' not in i and i not in ['x','y','holdout','city']]
    for indep_var in vars_indep:
        # calc max and min
        var_min = np.min(df[indep_var])
        var_max = np.max(df[indep_var])
        var_mean = np.mean(df[indep_var])
        var_sd = np.std(df[indep_var])
        normalize_parameters = normalize_parameters.append({'feature': indep_var,'mean': var_mean,'sd': var_sd,'max': var_max, 'min': var_min}, ignore_index=True)
        # transform to [0,1]
        # df[indep_var] = (df[indep_var] - var_min) / (var_max - var_min)
        df[indep_var] = (df[indep_var] - var_mean) / (var_sd)
    normalize_parameters.to_csv('data/normalization_parameters_{}.csv'.format(grid_size))
    return(df)


def holdout_grid(df_city, add_index):
    '''
    assign each row a spatial cell group number.
    holdouts will be done at the cell group to avoid overfitting
    '''
    # how many cells
    cell_size = 8
    coords = np.unique(df_city.x), np.unique(df_city.y)
    n_cells = round(len(coords[0]) / cell_size), round(len(coords[1]) / cell_size)
    # what are the coord cutoffs
    cutoffs_x = [coords[0][cell_size*i-1] for i in range(1,n_cells[0])] + [np.max(coords[0])]
    cutoffs_y = [coords[1][cell_size*i-1] for i in range(1,n_cells[1])] + [np.max(coords[1])]
    # assign a number to each of the rows as to which cell group it is in
    df_city['holdout'] = None
    for index, r in df_city.iterrows():
        r_ix = np.searchsorted(cutoffs_x,r.x)
        r_iy = np.searchsorted(cutoffs_y,r.y)
        df_city.loc[index, 'holdout'] = r_iy * n_cells[0] + r_ix

    # I want the holdout values to be unique (intercity)
    df_city.holdout += add_index
    add_index = np.max(df_city.holdout)
    return(df_city, add_index)

if __name__ == '__main__':
    main(scale)
