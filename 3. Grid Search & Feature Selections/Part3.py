import pandas as pd
import numpy as np
import warnings
import os
import pickle
import boto
import boto3
import time
import datetime
import sys
import luigi
import urllib

from boto.s3.key import Key
from datetime import datetime as dt
from sklearn.metrics import r2_score,mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler

global scaler
scaler = MinMaxScaler()


pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

class download_data(luigi.Task):
    global train, test
    def run(self):
        urllib.request.urlretrieve("https://s3.us-east-2.amazonaws.com/final-project-dataset/train.csv", filename='training_data.csv')
        urllib.request.urlretrieve("https://s3.us-east-2.amazonaws.com/final-project-dataset/test.csv", filename='testing_data.csv')
        train = pd.read_csv('training_data.csv')
        test = pd.read_csv('testing_data.csv')
        data_frame = train.append(test)

        data_frame.to_csv(self.output().path)

    def output(self):
        return luigi.LocalTarget('Combined_DF.csv')

class preprocess_data(luigi.Task):

    def requires(self):
        yield download_data()

    def run(self):
        data = pd.read_csv(download_data().output().path)

        # Combining Test and Train Frames
        #data = df_train.append(df_test)
        data.reset_index(inplace=True)
        data.drop('index', inplace=True, axis=1)

        # Deriving Time Series Columns from datetime field
        data["date"] = data.datetime.apply(lambda x: x.split()[0])
        data["hour"] = data.datetime.apply(lambda x: x.split()[1].split(":")[0]).astype("int")
        data["year"] = data.datetime.apply(lambda x: x.split()[0].split("-")[0]).astype('int')
        data["weekday"] = data.date.apply(lambda dateString: dt.strptime(dateString, "%Y-%m-%d").weekday())
        data["month"] = data.date.apply(lambda dateString: dt.strptime(dateString, "%Y-%m-%d").month)

        # Predicting Missing Wind Values using RF Regressor
        dataWind0 = data[data["windspeed"] == 0]
        dataWindNot0 = data[data["windspeed"] != 0]
        rfModel_wind = RandomForestRegressor()
        windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]
        rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])
        wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
        dataWind0["windspeed"] = wind0Values
        data = dataWindNot0.append(dataWind0)
        data.reset_index(inplace=True)
        data.drop('index', inplace=True, axis=1)

        # Designating Categorical Features from numeric columns
        categoricalFeatureNames = ["season", "holiday", "workingday", "weather", "weekday", "month", "year", "hour"]
        numericalFeatureNames = ["atemp", "humidity", "windspeed"]
        dropFeatures = ["casual", "datetime", "date", "registered", "temp"]

        dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
        dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

        dataTrain = dataTrain.drop(dropFeatures, axis=1)
        dataTest = dataTest.drop(dropFeatures, axis=1)
        dataTest = dataTest.drop('count', axis=1)

        dataTrain.to_csv(self.output().path)

    def output(self):
        return luigi.LocalTarget('preprocessed_train.csv')

class tune_model(luigi.Task):
    def requires(self):
        yield preprocess_data()

    def run(self):
        processed_train = pd.read_csv('preprocessed_train.csv')
        X = processed_train.drop('count', axis=1)
        Y = processed_train['count']

        #Since data is timeseries hence we MUST keep the shuffle parameter as FALSE !!
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=False)

        global error_metric
        error_metric = pd.DataFrame({'Training RMSLE': [],
                                     'Training R^2': [],
                                     'Testing RMSLE': [],
                                     'Testing R^2': []})

        def model_stats(model, model_name, X_train, Y_train, X_test, Y_test):
            global error_metric
            train_data_predictions = model.predict(X_train)
            test_data_predictions = model.predict(X_test)

            # RMSLE
            model_rmsle_train = np.sqrt(mean_squared_log_error(Y_train, train_data_predictions))
            model_rmsle_test = np.sqrt(mean_squared_log_error(Y_test, test_data_predictions))

            # R-Squared
            model_r2_train = r2_score(Y_train, train_data_predictions)
            model_r2_test = r2_score(Y_test, test_data_predictions)

            df_local = pd.DataFrame({'Model': [model_name],
                                     'Training RMSLE': [model_rmsle_train],
                                     'Training R^2': [model_r2_train],
                                     'Testing RMSLE': [model_rmsle_test],
                                     'Testing R^2': [model_r2_test]})

            error_metric = pd.concat([error_metric, df_local])

        def grid_Search(X_train, Y_train, X_test, Y_test):
            param_grid_rf = {'n_estimators': [5, 20, 100], 'max_depth': [5, 10, 50], 'oob_score': [True, False]}
            grid_rf = GridSearchCV(RandomForestRegressor(), param_grid=param_grid_rf, n_jobs=-1, cv=3, refit=True)
            grid_rf.fit(X_train, Y_train)
            print(grid_rf.best_params_)
            model_stats(grid_rf, 'Tuned RF Model', X_train, Y_train, X_test, Y_test)

        def get_scaled_data(X_train, X_test, Y_train, Y_test):
            # Scaling the data and checking with Tuned Model
            yte = np.array(Y_test).reshape(len(Y_test), 1)
            ytr = np.array(Y_train).reshape(len(Y_train), 1)

            X_test_scaled = scaler.fit_transform(X_test)
            X_train_scaled = scaler.fit_transform(X_train)

            Y_train_scaled = scaler.fit_transform(ytr).ravel()
            Y_test_scaled = scaler.fit_transform(yte).ravel()

            return X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled

        def boruta_feature_selection(X_train, Y_train, X_test, Y_test):
            X_Boruta = X_train.values
            Y_Boruta = Y_train.values
            Y_Boruta = Y_Boruta.ravel()

            tuned_model_boruta = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=False, n_jobs=-1)

            # define Boruta feature selection method
            feat_selector = BorutaPy(tuned_model_boruta, n_estimators='auto', random_state=1, max_iter=50)

            # find all relevant features
            feat_selector.fit(X_Boruta, Y_Boruta)

            boruta_features = ['weather', 'humidity', 'year', 'hour', 'month', 'workingday', 'season', 'atemp',
                               'weekday']

            X_train_Boruta = X_train[boruta_features]
            X_test_Boruta = X_test[boruta_features]

            boruta_model = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=False, n_jobs=-1)
            boruta_model.fit(X_train_Boruta, Y_train)
            model_stats(boruta_model, 'Boruta Model', X_train_Boruta, Y_train, X_test_Boruta, Y_test)

        def forward(X_train, Y_train):
            rf_sfs = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=False, n_jobs=-1)
            SFS = SequentialFeatureSelector(rf_sfs, k_features=6, scoring='neg_mean_squared_error', n_jobs=-1)
            SFS = SFS.fit(X_train, Y_train)
            print(SFS.k_feature_names_)

        def backward(X_train, Y_train):
            rf_sfs = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=False, n_jobs=-1)
            SFS_b = SequentialFeatureSelector(rf_sfs, forward=False, k_features=6, scoring='neg_mean_squared_error',
                                              n_jobs=-1)
            SFS_b = SFS_b.fit(X_train.values, Y_train.values)
            indxs = list(SFS_b.k_feature_names_)
            str_cols = X_train.columns
            features = set(zip(indxs, str_cols))
            print(features)

        print('Grid Search CV...')
        # Grid Search
        grid_Search(X_train, Y_train, X_test, Y_test)
        print('Done')
        print('\n')

        X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled = get_scaled_data(X_train, X_test, Y_train, Y_test)

        print('Scaled Data on the model...')
        # Scaled Data on Tuned Model
        scaled_rf = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=False, n_jobs=-1)
        scaled_rf.fit(X_train_scaled, Y_train_scaled)
        model_stats(scaled_rf, 'Scaled & Tuned RF', X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled)
        print('Done')
        print('\n')

        print('Boruta Feature Selection...')
        boruta_feature_selection(X_train, Y_train, X_test, Y_test)
        print('Done')
        print('\n')

        print('Forward Search...')
        # Sequential Forward Search
        forward(X_train, Y_train)
        print('Done')
        print('\n')

        print('Backward Search...')
        # Sequential Backward Search
        backward(X_train, Y_train)
        print('Done')
        print('\n')

        print('Filtered Features on Model')
        # Filtered Features After Searches
        rf_fil = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=False, n_jobs=-1)
        common_features = ['humidity', 'weather', 'season', 'atemp', 'workingday', 'hour', 'year', 'month']
        X_train_filtered = X_train[common_features]
        X_test_filtered = X_test[common_features]

        rf_fil.fit(X_train_filtered, Y_train)
        model_stats(rf_fil, 'Filtered Features RF', X_train_filtered, Y_train, X_test_filtered, Y_test)
        print('Done')
        print('\n')

        print('Pickling the Model')
        # Pickle the model
        finalized_model = open("Finalized_Model_RF.pkl", "wb")
        pickle.dump(rf_fil, finalized_model)
        finalized_model.close()
        print('Done')
        print('\n')

        error_metric.reset_index().drop('index', axis=1).to_csv(self.output().path)

    def output(self):
        return luigi.LocalTarget('Summary of Tuning.csv')

class upload_to_s3(luigi.Task):
    awsaccess = luigi.Parameter()
    awssecret = luigi.Parameter()
    inputLocation = luigi.Parameter()

    def requires(self):
        yield tune_model()

    def run(self):
        awsaccess = self.awsaccess
        awssecret = self.awssecret
        inputLocation = self.inputLocation

        # loaded_model = pickle.load(open('Finalized_Model_RF.pkl', 'rb'))

        try:
            conn = boto.connect_s3(awsaccess,awssecret)
            print("Connected to S3")
        except:
            print("Amazon keys are invalid!!")
            exit()

        loc = ''

        if inputLocation == 'APNortheast':
            loc = boto.s3.connection.Location.APNortheast
        elif inputLocation == 'APSoutheast':
            loc = boto.s3.connection.Location.APSoutheast
        elif inputLocation == 'APSoutheast2':
            loc = boto.s3.connection.Location.APSoutheast2
        elif inputLocation == 'CNNorth1':
            loc = boto.s3.connection.Location.CNNorth1
        elif inputLocation == 'EUCentral1':
            loc = boto.s3.connection.Location.EUCentral1
        elif inputLocation == 'EU':
            loc = boto.s3.connection.Location.EU
        elif inputLocation == 'SAEast':
            loc = boto.s3.connection.Location.SAEast
        elif inputLocation == 'USWest':
            loc = boto.s3.connection.Location.USWest
        elif inputLocation == 'USWest2':
            loc = boto.s3.connection.Location.USWest2

        try:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts)
            bucket_name = 'finalprojectpart3' + str(st).replace(" ", "").replace("-", "").replace(":", "").replace(".","")
            bucket = conn.create_bucket(bucket_name, location=loc)

            print("bucket created")
            s3 = boto3.client('s3', aws_access_key_id=awsaccess, aws_secret_access_key=awssecret)

            print('s3 client created')
            s3.upload_file(os.getcwd() +'\\'+'Summary of Tuning.csv', bucket_name, 'Summary of Tuning.csv')
            s3.upload_file(os.getcwd() +'\\'+ 'Finalized_Model_RF.pkl', bucket_name,'Finalized_Model_RF.pkl')

            print("File successfully uploaded to S3", 'Summary of Tuning.csv', bucket)
            print("File successfully uploaded to S3", 'Finalized_Model_RF.pkl', bucket)

        except:
            print("Amazon keys are invalid!!")
            exit()

if __name__ == '__main__':
    luigi.run()