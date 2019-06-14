# Grab AI for SEA - [Safety Challenge](https://www.aiforsea.com/safety)

## Table of Content

1. <a href='#1'> Overview of Solution </a>
2. <a href='#2'> Data Pre-processing </a>
3. <a href='#3'> How to Predict on New Dataset </a>

## <a id='1'> 1. Overview of Solution </a>

3 types of models are used in the final ensemble. As this challenge is essentially a time series classification, popular time series models like Long Short Term Memory (LSTM) and Convolutional Neural Network (CNN) which excel at capturing the temporal characteristics of the dataset were used. In addition, to introduce diversity to the ensemble, light gradient boosting machines (LGBM) was also attempted, using various statistics of the temporal features. 

![image](https://user-images.githubusercontent.com/43180977/59527572-8b3f3400-8f0e-11e9-8259-f702f43be42f.png)

Various configurations and hyperparameters of the base models were experimented (manual tuning for LSTM and CNN, automated Bayesian optimization for LGBM) using a stratified 5-fold cross validation scheme. The best performance (measured in terms of out-of-fold Area Under Receiver Operating Characteristic Curve - OOF AUROC) achieved by the individual models are as followed:
- LSTM - 0.7494
- CNN  - 0.7341
- LGBM - 0.7300

A linear regression model was fitted using the OOF predictions from the individual models, which determines the models' contributions to the ensemble. The ensemble improved the OOF AUROC to **0.7537**.

Model and ensemble details can be found in `development_notebooks/model_training`

## <a id='2'> 2. Data Pre-processing </a>

### Data Cleaning & Transformation

- Removed `bookingIDs` with duplicated labels
- Shifted bookingIDs' start time to 0 and adjusted subsequent time stamps accordingly
- Removed timestamps greater than 1800 seconds - 30 mins is more than enough to tell you if the driver is dangerous or not
- Clipped the maximum `Speed` to 35 m/s, which is equivalent to 126 km/h - The speed limit on most expressways in Singapore is only 90 km/h 
- Corrected the coordinates reference system (from phone to vehicle) using median values of `Acceleration X, Y, Z` and applied correction to `Acceleration X, Y, Z` and `Gyro X, Y, Z`. 
  This corrected features are more representative of the driver's behaviour (Z - vehicle bouncing, X - braking/ acceleration, Y - turning speed).
- Corrected the magnitude of `Acceleration X, Y, Z` by obtaining the G-multiplier on Z axis, using absolute gravitational force of 9.81 m/s^2. This helps to solve the issue where certain phones' accelerometer magnitude are 4x more than the grativational force.
- Rolling Mean of the sensor-based features (10 seconds rolling time frame) to smooth out the sensors' inherent inaccuracies.
- Rolling Max of the sensor-based features (10 seconds rolling time frame) to retain the spikes in sensors' values over the next 10 seconds.
- Null values between timestamps were interpolated linearly

### Feature Engineering & Selection

- `Turning Aggression` - The product of Angular Frequency and Speed, in log scale. The higher the combined magnitude of speed and angular frequency, the more aggressive the turn, as the driver did not slow down the vehicle while turning.
- `Magnitude of Acceleration X & Y` - Similar to turning aggression, but measured by the accelerometer.
- `Acceleration (derived)` - The acceleration calculated using vehicle Speed over recorded time stamp.
- Feature statistics - The mean, median, max, skewness and kurtosis of the selected features (group by bookingID) were generated.
- Feature selection - Iterative elimination and addition of features were conducted with 4 runs of base model (LSTM) which provides a quick gauge on the features performance. Features that led to model deterioration or insignificant improvements were dropped. The complete set of model performance and features configurations is available as an excel sheet in `development_notebooks/model_training/feature_selection_using_lstm_baseline.xlsx`

### Features/ Approaches explored but not used (led to deterioration/ insignificant improvements)
- Correction of `Acceleration X and Y`. This is because the previous XYZ correction results in indeterminate direction of the corrected X and Y (resultant X and Y are orthogonal to Z, but might not be pointing to perpendicular or parallel to the direction of vehicle travel). 
- Feature - `Use Phone While Driving`. This is a boolean feature where 1 signifies driver used phone while driving and 0 signifies driver did not use phone while driving. It is derived from the corrected `Acceleration Z`, where the Z value is less than 7 m/s^2 or more than 13 m/s^2 when Speed is non-zero. The high change in Z value happens when phones are moved from their stable position (phone stand) and signifies that the driver is using phone when driving.
- Feature - `Using Phone`. Similar to the above, but it is tied to the time stamp, 1 signifies driver using phone at the timestamp, 0 otherwise.
- Feature - `Jerk Rate`. This is the 2nd derivative of Speed or the rate of change of acceleration. High jerk rate typically signifies unstable ride resulted from sudden change of direction, braking or acceleration.
- Feature - `Angular Frequency`. This is the 1st derivative of Bearing or the rate of change of bearing direction. High angular frequency is associated with sudden turns.


### Data Standardization & Training Scheme

Data were split into the 5 folds using stratified sampling (for stratified 5-fold cross validation). For each fold, a standardization scaler was fitted using the training data, before the fitted means and scales were applied on the validation data. This ensures no data leakage to the validation set. For ease of loading and model training, each of the data are prepared and stored separately.

Complete preprocessing pipeline and data preparation can be found in `development_notebooks/feature_preprocessing_engineering/`

## <a id='3'> 3. How to Predict on New Dataset </a>

- Step 1 : Clone the repository.

- Step 2 : Create a new environment on Anaconda (alternatively on VirtualEnv) through command line. 

  `conda create --name your_env_name python=3.6.6`

- Step 3 : Navigate to where the repository is located and activate the newly created environment.

  On Windows -
  `activate your_env_name`

  On Linux -
  `source activate your_env_name`

- Step 4 : Install all the required dependencies.

  `pip install -r requirements.txt`

- Step 5 : Create a new folder named 'new_data' and move the prediction files (.csv) to the folder.

- Step 6 : Run the Python file in the base directory and the predictions will be saved on your base directory as 'ensemble_predictions.csv'

  `python preprocess_and_predict.py`
  
  #### Prepared by Chew Ze Yong
  Email : zychew1@e.ntu.edu.sg
