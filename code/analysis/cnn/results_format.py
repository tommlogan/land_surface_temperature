import pandas as pd

responses = ['lst_day_mean', 'lst_night_mean','lst_day_max', 'lst_night_max']

def pdp_results(grid_size):
    '''
    Format the partial dependence results data
    '''
    results_partial = pd.DataFrame()
    # import normalization data
    normalize_parameters = pd.read_csv('data/normalization_parameters_{}.csv'.format(grid_size))
    normalize_parameters = normalize_parameters.set_index('feature')

    # import results
    for h in responses:
        result_cnn = pd.read_csv('data/cnn/partial_dependence_{}_{}.csv'.format(grid_size, h))
        cols = result_cnn.columns.tolist()
        cols.remove('i_holdout')
        cols.remove('x_index')
        features = list(sorted(set(c[2:] for c in cols)))
        # loop through each feature
        for feature in features:
            df_feature = result_cnn[['x_' + feature, 'y_' + feature, 'i_holdout']]
            # unnormalize the feature
            df_feature['x_' + feature] = df_feature['x_' + feature].copy()*normalize_parameters.loc[feature,'sd'] + normalize_parameters.loc[feature,'mean']
            # rename columns
            df_feature['x'] = df_feature['x_' + feature]
            df_feature['mean'] = df_feature['y_' + feature]
            df_feature['boot'] = df_feature['i_holdout']
            df_feature = df_feature.drop(['x_{}'.format(feature), 'y_{}'.format(feature), 'i_holdout'],axis=1)
            # put the new ones in
            df_feature['model'] = 'cnn'
            df_feature['dependent'] = h
            df_feature['independent']= feature
            results_partial = results_partial.append(df_feature, ignore_index=True)
    # append to existing results
    # import results
    results_pd = pd.read_csv('data/regression/results_partial_dependents_{}.csv'.format(grid_size))
    # concatenate
    results_pd = pd.concat([results_pd, results_partial], ignore_index=True)
    # save results
    results_pd.to_csv('data/regression/results_partial_dependents_{}_all.csv'.format(grid_size))


def varimp_results(grid_size):
    '''
    variable importance results
    '''
    results_swing = pd.read_csv('data/regression/results_swing_{}.csv'.format(grid_size))

    for h in responses:
        results_cnn = pd.read_csv('data/cnn/swing_{}_{}.csv'.format(grid_size, h))
        # update column name
        results_cnn['raw'] = results_cnn.swing
        # calculate the swing
        total_range = results_cnn.raw.sum()
        results_cnn['swing'] = results_cnn.raw/total_range
        # add model and dependent column
        results_cnn['model'] = 'cnn'
        results_cnn['dependent'] = h
        results_cnn['independent'] = results_cnn.feature
        results_cnn = results_cnn.drop(['feature'],axis=1)
        # save
        results_swing = results_swing.append(results_cnn, ignore_index=True)
        # save results
        results_swing.to_csv('data/regression/results_swing_{}.csv'.format(grid_size))


def holdout_results(grid_size):
    '''
    reformat the holdout results from CNN
    '''
    time_of_day = {'lst_day_mean':'diurnal', 'lst_night_mean':'nocturnal',
                    'lst_day_max':'diurnalmax', 'lst_night_max':'nocturnalmax'}
    # loop dependent variables
    for h in responses:
        results_cnn = pd.read_csv('data/cnn/metrics_{}_{}.csv'.format(grid_size, h))
        results_cnn = pd.melt(results_cnn, id_vars = ['i_holdout'],
                            var_name = 'error_metric', value_name = 'error')
        results_cnn['hold_num'] = results_cnn.i_holdout
        results_cnn = results_cnn.drop(['i_holdout'], axis=1)
        results_cnn['time_of_day'] = time_of_day[h]
        results_cnn['model'] = 'cnn'
        results_cnn.to_csv('data/regression/holdout_{}/cnn_{}.csv'.format(grid_size,h))



if __name__ == '__main__':
    # profile() # initialise the board
    grid_size = 500
    pdp_results(grid_size)
    varimp_results(grid_size)
    holdout_results(grid_size)
