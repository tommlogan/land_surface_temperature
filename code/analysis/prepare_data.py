'''
Integrate the data sets, and scale the variables.
Output is a csv with data for analysis
'''

# import libraries
import pandas as pd
import numpy as np

def main():
    
    cities = ['bal','det','phx','por']
    # init all city dataframe
    df = pd.DataFrame()
    for city in cities:
        # import city data
        df_city = pd.read_csv('data/processed/grid/{}/2018-04-14/{}_data.csv'.format(city,city))
        # scale city specific variables
        df_city = scaling_city(df_city)
        # add the grid group numbers
        
        # append city name to df
        df_city['city'] = city
        # bind to complete df
        df = df.append(df_city, ignore_index=True)
    # apply transformation on entire dataset
    df = scaling_all(df)
    # write to csv
    df.to_csv('data/data_regressions.csv')


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
    df_city = df_city.drop(['ntl_sd','ntl_sd_sl'], axis=1)
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
    df_city[vars_elev] = df[vars_elev] - elev_means
    
    ### DSM ###
    # subtract the mean from all of the dsm variables (except standard deviation)
    # list dsm vars
    vars_dsm = [i for i in vars_all if 'dsm' in i]
    # remove sd vars
    vars_dsm = [x for x in vars_dsm if 'sd' not in x]
    # calculate the mean of each
    dsm_means = df_city[vars_dsm].mean(axis=0)
    df_city[vars_dsm] = df[vars_dsm] - dsm_means
    
    ### LST ###
    # drop the lst vars except night and day mean
    vars_response = ['lst_day_mean','lst_night_mean']
    vars_lst = [i for i in vars_all if 'lst' in i]
    # remove vars to keep
    vars_lst_drop = [x for x in vars_lst if x not in vars_response]
    # drop vars
    df = df.drop(vars_lst_drop, axis=1)
    # calculate the mean of each
    lst_means = df_city[vars_response].mean(axis=0)
    df_city[vars_response] = df[vars_response] - lst_means
    
    return(df_city)

    
def scaling_all(df):
    '''
    scale the variables between the cities
    '''
    
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
    df = df.drop(vars_lcov, axis=1)
    
    ### Building area
    # set the building area as a percent of the cell's area
    df.bldg = df.bldg/df.area 
    
    # Transform to [0,1]
    vars_indep = [i for i in vars_all if 'lst' not in i]
    for indep_var in vars_indep:
        # calc max and min
        var_min = np.min(df[indep_var])
        var_max = np.max(df[indep_var])
        # transform to [0,1]
        df[indep_var] = (df[indep_var] - var_min) / (var_max - var_min)
    
    return(df)