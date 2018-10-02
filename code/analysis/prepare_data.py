'''
Integrate the data sets, and scale the variables.
Output is a csv with data for analysis
'''

# import libraries
import pandas as pd
import numpy as np

def main():

    cities = ['bal','det','phx','por']
    grid_size = 500
    # init all city dataframe
    df = pd.DataFrame()
    for city in cities:
        # import city data
        df_city = pd.read_csv('data/processed/grid/{}/2018-10-01/{}_data_{}.csv'.format(city,city,grid_size))
        # scale city specific variables
        df_city = scaling_city(df_city)
        # add the grid group numbers
        df_city = holdout_grid(df_city)
        # append city name to df
        df_city['city'] = city
        # bind to complete df
        df = df.append(df_city, ignore_index=True)
    # apply transformation on entire dataset
    df = scaling_all(df)
    # write to csv
    df.to_csv('data/data_regressions_{}.csv'.format(grid_size))


def scaling_city(df_city):
    '''
    scale the variables which are city specific (e.g. elevation has the city mean removed before [0,1] scaling)
    '''

    vars_all = df_city.columns.values

    ### ADMIN ###
    df_city = df_city.drop(['Unnamed: 0', 'cId'], axis = 1)
    # outside city limits
    df_city = df_city[df_city['area'] > 0]

    ### NaN variables
    # remove night time lights (not enough variance to get standard deviation)
    vars_ntl = [x for x in vars_all if 'ntl' in x]
    df_city = df_city.drop(vars_ntl, axis=1)
    # drop any nan rows and report how many dropped
    df_city = df_city.dropna(axis=0, how='any')

    ### Elevation ###
    # subtract the mean from all of the elevation variables (except standard deviation)
    # list elev vars
    vars_elev = [i for i in vars_all if 'elev' in i]
    # remove sd vars
    vars_elev = [x for x in vars_elev if 'sd' not in x]
    # calculate the mean of each
    elev_means = df_city[vars_elev].mean(axis=0)
    df_city[vars_elev] = df_city[vars_elev] - elev_means



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
    vars_response = ['lst_day_mean','lst_night_mean']
    vars_lst = [i for i in vars_all if 'lst' in i]
    # remove vars to keep
    vars_lst_drop = [x for x in vars_lst if x not in vars_response]
    # drop vars
    df_city = df_city.drop(vars_lst_drop, axis=1)
    # calculate the mean of each
    lst_means = df_city[vars_response].mean(axis=0)
    df_city[vars_response] = df_city[vars_response] - lst_means

    return(df_city)


def scaling_all(df):
    '''
    scale the variables between the cities
    '''
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
        df[var_lcov] = df[var_lcov]/area_max
    # drop rows with water more than 20% of area
    df = df.loc[df['lcov_11'] < 0.2]
    # drop lcov variables
    vars_lcov = [i for i in df.columns.values if 'lcov' in i]
    # keep water (lcov_11)
    vars_lcov = [i for i in vars_lcov if '11' not in i]
    df = df.drop(vars_lcov, axis=1)

    ### Drop impervious surface
    vars_imp = [i for i in vars_all if 'imp' in i]
    df = df.drop(vars_imp, axis=1)

    ### Building area
    # set the building area as a percent of the cell's area
    df.bldg = df.bldg/df.area

    # drop area
    df = df.drop('area', axis=1)

    # Transform to [0,1]
    vars_all = df.columns.values
    vars_indep = [i for i in vars_all if 'lst' not in i and i not in ['x','y','holdout','city']]
    for indep_var in vars_indep:
        # calc max and min
        var_min = np.min(df[indep_var])
        var_max = np.max(df[indep_var])
        # transform to [0,1]
        df[indep_var] = (df[indep_var] - var_min) / (var_max - var_min)

    return(df)


def holdout_grid(df_city):
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

    return(df_city)

if __name__ == '__main__':
    main()
