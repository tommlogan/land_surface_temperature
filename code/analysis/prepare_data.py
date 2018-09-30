'''
Integrate the data sets, and scale the variables.
Output is a csv with data for analysis
'''

# import libraries
import pandas as pd


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
    scaling_all(df)


def scaling_city(df_city):
    '''
    scale the variables which are city specific (e.g. elevation has the city mean removed before other scaling
    '''
    
    ### ADMIN ###
    df = df.drop(['Unnamed: 0', 'cId'], axis = 1)
    # outside city limits
    df = df[df['area'] > 0]
    
    ### NaN variables
    # remove night time lights (not enough variance to get standard deviation)
    df = df.drop(['ntl_sd','ntl_sd_sl'], axis=1)
    # drop any nan rows and report how many dropped
    df = df.dropna(axis=0, how='any')
    
    ### Elevation ###
    
    
    ### DSM ###
    
    
    ### LST ###
    

    
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
    
    ### Tree canopy ###
    
    
    
    ### Standard Deviation ###