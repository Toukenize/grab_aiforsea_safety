import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from sklearn.metrics import roc_auc_score
from keras.models import load_model
from keras import backend as K

def load_all_csv():

    """
    Load all the features csv files, keep the desired columns, and sort by bookingID and second.
    """
    
    paths = glob('new_data/*.csv')
    
    assert len(paths) > 0, 'Please move all the new data to the folder "new_data" and try again'
    
    features = pd.read_csv(paths[0])
    
    for path in tqdm(paths[1:]):
        new_df = pd.read_csv(path)
        features = pd.concat([features, new_df])

    # check that the features are present and in sequence
    try:
        features = features[[   
                                'bookingID',
                                'Accuracy',
                                'Bearing',
                                'acceleration_x',
                                'acceleration_y',
                                'acceleration_z',
                                'gyro_x',
                                'gyro_y',
                                'gyro_z',
                                'second',
                                'Speed'
                            ]]
    except KeyError:
        print('One or more feature columns not found. Please ensure only csv files with the exact columns are in the directory')
    
    features.drop(columns=['gyro_x','gyro_y','gyro_z'], inplace=True)
    features.sort_values(by=['bookingID','second'], inplace=True)
    features.set_index(['bookingID','second'], inplace=True)
    
    if not (~features.isna()).all().all():
        # interpolate dataframe if there are missing numerical data
        features.interpolate(inplace=True)
        
    return features

def get_booking_details(dataframe):

    """
    Utility function to obtain the booking start index, end index and total length (rows of data associated with the booking), for optimized feature engineering.
    """
    bking_index = dataframe.index.get_level_values(0).unique()
    bking_index_start = dataframe[['Accuracy']].groupby('bookingID').count().cumsum().shift(1).values.ravel()
    bking_index_start[0] = 0
    bking_index_start = bking_index_start.astype(np.int64)
    bking_index_end = dataframe[['Accuracy']].groupby('bookingID').count().cumsum().values.ravel()
    bking_index_details = dict((k, (start, end)) for k, start, end in \
                              zip(bking_index, bking_index_start, bking_index_end))
    
    return bking_index_details, len(bking_index)

def get_rotation_matrix_3d(i_v, unit=None):
    
    """
    Obtain the 3d rotation matrix that rotates the vector to maximise its magnitude in a single axis. This is used to re-orientate the phone's coordinate system to the absolute coordinate system of the vehicle.
    
    Code reference : https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis
    """
    
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

def get_df_3d_matrix_g_multiplier(dataframe, num_booking):

    """
    Obtain the 3d rotation matrix for each booking, and the g multiplier needed to restore the z-axis magnitude to 9.81 (gravitational force)
    """
    
    acc_medians = dataframe[['acceleration_x',\
                             'acceleration_y',\
                             'acceleration_z']].groupby(by='bookingID').median()
    
    acc_medians['acc_mag'] = acc_medians.apply(lambda x : np.sqrt(x['acceleration_x'] ** 2 + \
                                                                  x['acceleration_y'] ** 2 + \
                                                                  x['acceleration_z'] ** 2), axis=1)
    
    g_multiplier = acc_medians['acc_mag'].values / 9.81
    
    acc_medians_values = acc_medians[['acceleration_x',\
                                      'acceleration_y',\
                                      'acceleration_z']].values
    rotation_matrix_3d = np.zeros((num_booking, 3, 3))
    for i, acc_val in enumerate(tqdm(acc_medians_values)):
        rotation_matrix_3d[i] = get_rotation_matrix_3d(acc_val)
    
    return rotation_matrix_3d, g_multiplier

def correct_xyz(dataframe, bking_details, rotation_matrix_3d, g_multiplier):

    """
    Apply the rotation matrix to the bookingID's xyz accelerations entry, on the assumption that the phone orientation within the vehicle remains constant throughout the journey.
    """
    
    acc_original = dataframe[['acceleration_x', 'acceleration_y','acceleration_z']].values
    acc_corrected = np.zeros_like(acc_original)

    for b_index, (rot, g, (start, end)) in enumerate(tqdm(zip(rotation_matrix_3d, g_multiplier, bking_details.values()), total=len(g_multiplier))):
        for r_index, row in enumerate(acc_original[start:end], start):
            acc_corrected[r_index] = np.dot(row.T, rot.T) * g

    dataframe[['acceleration_z', 'acceleration_x','acceleration_y']] = acc_corrected
    
    return dataframe

def shift_start_time(dataframe, bking_details):

    """
    Shift the time with respect to the earliest timestamp.
    """
    bkingid_start_index = [start for start, _ in bking_details.values()]
    
    dataframe.reset_index(inplace=True)
    zerolised_time = dataframe['second'].values
    for start, end in tqdm(bking_details.values()):
        bking_start_time = dataframe.iloc[start, dataframe.columns.get_loc('second')]
        zerolised_time[start : end] -= bking_start_time
    dataframe['second'] = zerolised_time
    
    dataframe.drop(labels = dataframe.loc[dataframe['second'] >= 1800,:].index, inplace=True)
    
    dataframe.set_index(['bookingID','second'], inplace=True)
    
    return dataframe

def interpolate_time(dataframe, bking_details):
    
    """
    Interpolate missing time and features in between entries.
    """
    
    dataframe.reset_index(inplace=True)
    
    bking_interpolate_list = dict()
    
    for bkingid, (start, end) in tqdm(bking_details.items()):
        unique_time = set(dataframe.iloc[start:end, dataframe.columns.get_loc('second')])
        full_time_list = set(range(int(min(unique_time)), int(max(unique_time)) + 1, 1))
        bking_interpolate_list[bkingid] = full_time_list.difference(unique_time)
    
    bking_interpolate_len = dict((k, len(v)) for k, v in bking_interpolate_list.items())
    
    row_num = sum(bking_interpolate_len.values())
    col_num = len(dataframe.columns)
    
    interpolate_arr = np.empty((row_num, col_num))
    interpolate_arr[:] = np.nan
    
    count = 0
    
    for bkingid, missing_time in tqdm(bking_interpolate_list.items()):
        for time in missing_time:
            interpolate_arr[count][dataframe.columns.get_loc('bookingID')] = bkingid
            interpolate_arr[count][dataframe.columns.get_loc('second')] = time
            count += 1
    
    interpolate_df = pd.DataFrame(interpolate_arr, columns=dataframe.columns)
    
    dataframe = pd.concat([dataframe, interpolate_df])
    dataframe['bookingID'] = dataframe['bookingID'].astype('int64')
    dataframe.sort_values(by=['bookingID','second'], inplace=True)
    dataframe.set_index(['bookingID','second'], inplace=True)
    
    dataframe.interpolate(inplace=True)
    
    return dataframe
    
def get_engineered_features(dataframe, bking_details):
    
    """
    Correct speed more than 35 m/s, convert bearing to radians, and generate desired features:
    1. Acceleration -> Derivative of Speed
    2. Turning Aggression -> log(| AngularFrequency * Speed |)
    3. XY Acceleration Magnitude -> sqrt(AccelerationX ** 2 + AccelerationY **2)
    """
    
    start_idx = [x[0] for x in bking_details.values()]
    
    # remove speed outliers (more than 35 m/s is unlikely in Singapore)
    dataframe.loc[dataframe['Speed'] > 35, 'Speed']  = 35
    
    dataframe['Acc_derived'] = (dataframe['Speed'] - dataframe['Speed'].shift(1))
    dataframe.iloc[start_idx, dataframe.columns.get_loc('Acc_derived')] = 0

    #dataframe['Jerk_derived'] = (dataframe['Acc_derived'] - dataframe['Acc_derived'].shift(1))
    #dataframe.iloc[start_idx, dataframe.columns.get_loc('Jerk_derived')] = 0

    # change bearing to radians
    dataframe['Bearing'] = dataframe['Bearing'].apply(lambda x : np.deg2rad(x))
    dataframe['Angular_freq_derived'] = dataframe['Bearing'] - dataframe['Bearing'].shift(1)
    dataframe.iloc[start_idx, dataframe.columns.get_loc('Angular_freq_derived')] = 0
        
    # get turning aggression - ang freq * speed
    dataframe['Turning_aggression'] = np.log1p(np.abs(dataframe['Angular_freq_derived'] * dataframe['Speed']))
    
    dataframe['acc_xy_mag'] = np.sqrt(dataframe['acceleration_x'] ** 2 + dataframe['acceleration_y'] ** 2)

    dataframe.drop(columns=['Bearing','Angular_freq_derived','acceleration_x','acceleration_y'], inplace=True)
    
    return dataframe

def get_moving_stat(dataframe, cols, window, bking_details, stat):
    
    """
    Generate the moving statistics of the selected columns.
    """
    
    for col in cols:
        if col not in dataframe.columns:
            print(f'{col} not in dataframe, please check')
            return None
    
    new_cols = [col + f'_mvg_{stat}_{window}' for col in cols]
    
    for col in new_cols:
        dataframe[col] = 0
    
    if stat == 'mean':
        new_cols_values = dataframe[cols].rolling(window = window).mean().values
    elif stat == 'max':
        new_cols_values = dataframe[cols].rolling(window = window).max().values
        
    for start, _ in tqdm(bking_details.values()):
        new_cols_values[start : start + window] = 0
    
    dataframe[new_cols] = new_cols_values
    
    return dataframe

def generate_feature_statistics(dataframe):
    
    """
    Generate feature statistics of the processed features set.
    """
    
    features_medians = dataframe.groupby('bookingID').median()
    features_max = dataframe.groupby('bookingID').max()
    features_mean = dataframe.groupby('bookingID').mean()
    features_skew = dataframe.groupby('bookingID').skew()
    features_kurt = dataframe.groupby('bookingID').apply(pd.DataFrame.kurt)
    
    features_medians.columns = [x + '_median' for x in features_medians.columns]
    features_max.columns = [x + '_max' for x in features_max.columns]
    features_mean.columns = [x + '_mean' for x in features_mean.columns]
    features_skew.columns = [x + '_skew' for x in features_skew.columns]
    features_kurt.columns = [x + '_kurt' for x in features_kurt.columns]
    
    feature_stats = pd.concat([features_medians, features_max, features_mean, features_skew, features_kurt], axis=1)
    
    values = feature_stats.values
    values[np.isnan(values)] = 0
    
    feature_stats[feature_stats.columns] = values
    
    return feature_stats
   
def parallelize_dataframe(df_split, func, num_workers):

    """
    Utility function to parallelize the preprocessing of dataframe using multiprocessing.
    """
    pool = Pool(num_workers)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def get_all_features(features):
    
    """
    Obtain various features and update the features dataframe
    """
    
    bking_index_details, num_bkings = get_booking_details(features)

    # correct xyz coordinates
    rot3d, gmul = get_df_3d_matrix_g_multiplier(features, num_bkings)    
    features = correct_xyz(features, bking_index_details, rot3d, gmul)
    
    # shift and interpolate
    features = shift_start_time(features, bking_index_details)
    bking_index_details, num_bkings = get_booking_details(features)
    features = interpolate_time(features, bking_index_details)
    bking_index_details, num_bkings = get_booking_details(features)
    
    # get engineered features
    features = get_engineered_features(features, bking_index_details)
    
    # get moving stats
    cols_for_mvg_stat = ['acceleration_z', 'Speed', 'Acc_derived', 'acc_xy_mag']
    features = get_moving_stat(features, cols_for_mvg_stat, 10, bking_index_details, 'mean')
    features = get_moving_stat(features, cols_for_mvg_stat, 10, bking_index_details, 'max')
    cols_for_mvg_stat = ['Turning_aggression']
    features = get_moving_stat(features, cols_for_mvg_stat, 3, bking_index_details, 'mean')
    
    features.drop(columns=['Turning_aggression','acc_xy_mag','acc_xy_mag_mvg_mean_10'], inplace=True)
    if not (~features.isna()).all().all():
        features.interpolate(inplace=True)
        
    return features

def get_dataframe_partitions(features, num_partition):
    
    """
    Partition dataframe into smaller partitions, to enable multiprocessing
    """
    
    bking_index_details, num_bkings = get_booking_details(features)
    bookingIDs = np.array(features.index.get_level_values(0).unique())
    partition = []
    partition_size = int(bookingIDs.shape[0] / num_partition)
    for i in range(num_partition):
        start_id =  bookingIDs[partition_size * i]
        if i + 1 == num_partition:
            # to cater for edge cases
            end_id = bookingIDs[-1]
        else:
            end_id = bookingIDs[partition_size * (i+1) - 1]
        partition.append((bking_index_details[start_id][0], 
                        bking_index_details[end_id][1]))  
    features_partitioned = [features.iloc[start:end,:] for start,end in partition]
    return features_partitioned

def preprocess_and_save_data():

    """
    Preprocess data and save the processed features and features statistics
    """
    
    print('Step 1 / 4 : Loading all the .csv files.. This took around 1 min on the given dataset')
    features = load_all_csv()
    features = features.astype(np.float32)
    
    num_partition = cpu_count()
    print('Step 2 / 4 : Generating features... This took around 2 mins on 4 workers for the given dataset.')
    print(f'Parallelizing jobs, using {num_partition} workers')
    features_partitioned = get_dataframe_partitions(features, num_partition)
    features = parallelize_dataframe(features_partitioned, get_all_features, num_partition)
    
    features_partitioned = get_dataframe_partitions(features, num_partition)
    feature_statistics = parallelize_dataframe(features_partitioned, generate_feature_statistics, num_partition)

    print('Step 3 / 4 : Saving features and features_stats')
    features.to_hdf('preprocessed_features.h5', key='features')
    feature_statistics.to_hdf('feature_stats.h5', key='feature_stats')
    
    bookingIDs = np.array(feature_statistics.index)
    
    return bookingIDs
    
def prepare_models_input(features, feature_stats, model_number):
    
    """
    Scale the data and convert the data according to model requirements (e.g. Neural network requires padded input sequence.)

    Note: As each fold were trained on different dataset, their respective feature means and scale are different. 
    This function prepares the data according to the specific feature means and scales used to train the particular Fold.
    """
    
    try:
        feature_scale = np.load(f'data_scale_mean/Fold{model_number}_train_data_scale.npy')
        feature_mean = np.load(f'data_scale_mean/Fold{model_number}_train_data_mean.npy')
        feature_stat_scale = np.load(f'data_scale_mean/Fold{model_number}_train_data_stats_scale.npy')
        feature_stat_mean = np.load(f'data_scale_mean/Fold{model_number}_train_data_stats_mean.npy')
    except IOError:
        print('Please ensure all 20 of the data scale and mean .npy files are in the folder "data_scale_mean"')
    features = (features - feature_mean) / feature_scale
    feature_stats = (feature_stats.values - feature_stat_mean) / feature_stat_scale
    
    feature_values = np.zeros((len(bookingIDs),1800,features.shape[1]), dtype=np.float32)
    features = np.array(features.groupby('bookingID').progress_apply(lambda x : x.values).values)
    for index, x in enumerate(tqdm(features)):
        end = x.shape[0]
        if end >= 1800:
            feature_values[index] = x[:1800]
        else:
            feature_values[index][:end] = x
            
    return feature_values, feature_stats    

def ensemble_predict(bookingIDs):
    
    """
    Generate predictions using the ensemble and save it as a .csv file in the same directory. 
    
    Ensemble members:
    1. 5-folds Long Short Term Memory Networks
    2. 5-folds Convolutional Neural Networks
    3. 5-folds Light Gradient Boosting Trees
    4. Weighted sum of the above, with the coefficients from a linear regression model fitted on the out-of-fold predictions of the above models.
    """
    
    print(f'Step 4 / 4 : Making Predictions using Ensemble and generating "ensemble_predictions.csv", estimated run time ~ 3 mins')
    for model_fold_number in range(1,6,1):
        print(f'Predicting - Fold {model_fold_number} / 5')
        
        # Loading the created features files
        try:
            features = pd.read_hdf('preprocessed_features.h5',key='features')
            features_stats = pd.read_hdf('feature_stats.h5',key='feature_stats')
        except IOError:
            print('Please check that the features files "preprocessed_features.h5" and "feature_stats.h5" are in the current directory')
        features, features_stats = prepare_models_input(features, features_stats, model_fold_number)
        
        # Prediction using LSTM
        try:
            lstm_model = load_model(f'lstm/lstm_fold{model_fold_number}_best_weights.hdf5', {'auroc':auroc})
        except IOError:
            print(f'Please ensure the model file "lstm_fold{model_fold_number}_best_weights.hdf5" is in the "lstm" folder')
        lstm_pred = lstm_model.predict(features, batch_size=128, verbose=1)
        K.clear_session()
        
        # Prediction using CNN
        try:
            cnn_model = load_model(f'cnn/cnn_fold{model_fold_number}_best_weights.hdf5', {'auroc':auroc})
        except IOError:
            print(f'Please ensure the model file "cnn_fold{model_fold_number}_best_weights.hdf5" is in the "cnn" folder')
        cnn_pred = cnn_model.predict(features, batch_size=128, verbose=1)
        K.clear_session()
        
        # Prediction using LGBM
        try:
            lgbm_model = lgb.Booster(model_file=f'lgbm/lgb_fold{model_fold_number}.txt')
        except IOError:
            print(f'Please ensure the model file "lgb_fold{model_fold_number}.txt" is in the "lgbm" folder')
        lgbm_pred = lgbm_model.predict(features_stats)

        if model_fold_number == 1:
            lstm_avg_pred = lstm_pred
            cnn_avg_pred = cnn_pred
            lgbm_avg_pred = lgbm_pred.reshape((lgbm_pred.shape[0],1))
        else:
            lstm_avg_pred += lstm_pred
            cnn_avg_pred += cnn_pred
            lgbm_avg_pred += lgbm_pred.reshape((lgbm_pred.shape[0],1))

    lstm_avg_pred /= 5
    cnn_avg_pred /= 5
    lgbm_avg_pred /= 5
    
    CNN_weight =  0.21416430
    LGBM_weight = 0.20759960
    LSTM_weight =  0.62418445

    ensemble_predictions = (CNN_weight * cnn_avg_pred + 
                            LGBM_weight * lgbm_avg_pred + 
                            LSTM_weight * lstm_avg_pred) / (CNN_weight + LGBM_weight + LSTM_weight)
    
    print(ensemble_predictions.shape)
    
    predictions_df = pd.DataFrame({'bookingID':bookingIDs})
    predictions_df['predictions'] = ensemble_predictions
    predictions_df.to_csv('ensemble_predictions.csv',index=False)
    
    return predictions_df
    
def roc_auc_score_modified(y_true, y_pred):
    
    """
    Modified ROC AUC Scoring method used to train Neural Networks
    - To tackle problems where shuffled batches only contains 1 class
    - Return 0.5 when this happens, an underestimate of the actual metric
    """
    
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5

def auroc(y_true, y_pred):
    
    """
    Custom Metric for Keras Neural Network - Area Under ROC Curve
    """
    
    return tf.py_function(roc_auc_score_modified, (y_true, y_pred), tf.double)
    
if __name__ == "__main__":
    tqdm.pandas()
    bookingIDs = preprocess_and_save_data()
    predictions = ensemble_predict(bookingIDs)