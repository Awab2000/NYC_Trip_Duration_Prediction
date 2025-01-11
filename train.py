import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


def compute_distance_from_lat_long(df_lat1, df_lon1, df_lat2, df_lon2):
    '''
    This function compute the distance between two pairs of (latitude,longitude)
    parameters: first location latitude, first location longitude, second location latitude, second location longitude
    returns distance
    '''
    # Approximate radius of earth in km
    R = 6373.0
    lat1 = np.radians(df_lat1)
    lon1 = np.radians(df_lon1)
    lat2 = np.radians(df_lat2)
    lon2 = np.radians(df_lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the compass bearing between two latitude/longitude points.

    Parameters:
        lat1, lon1: Latitude and longitude of the first point (in decimal degrees)
        lat2, lon2: Latitude and longitude of the second point (in decimal degrees)

    Returns:
        Compass bearing in degrees (0° = North, 90° = East, 180° = South, 270° = West)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the difference in longitude
    dlon = lon2 - lon1

    # Calculate the bearing
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(x, y)

    # Convert from radians to degrees and normalize to 0–360°
    compass_bearing = (np.degrees(initial_bearing) + 360) % 360

    return compass_bearing


def do_pca(df1, df2):
    coords = np.vstack((df1[['pickup_latitude', 'pickup_longitude']].values,
                        df1[['dropoff_latitude', 'dropoff_longitude']].values,
                        df2[['pickup_latitude', 'pickup_longitude']].values,
                        df2[['dropoff_latitude', 'dropoff_longitude']].values))

    pca = PCA().fit(coords)
    return pca, coords


def do_kmeans(coords):
    sample_ind = np.random.permutation(len(coords))[:1000000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
    return kmeans


def add_kmeans_features(df, kmeans):
    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])



def add_remove_noise_features(df):
    df['center_latitude'] = (df['pickup_latitude'] + df['dropoff_latitude']) / 2
    df['center_longitude'] = (df['pickup_longitude'] + df['dropoff_longitude']) / 2
    df['pickup_lat_bin'] = np.round(df['pickup_latitude'], 2)
    df['pickup_long_bin'] = np.round(df['pickup_longitude'], 2)
    df['center_lat_bin'] = np.round(df['center_latitude'], 2)
    df['center_long_bin'] = np.round(df['center_longitude'], 2)


def add_pca_features(df, pca):
    df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    df['pca_manhattan'] = np.abs(df['dropoff_pca1'] - df['pickup_pca1']) + np.abs(df['dropoff_pca0'] -
                                                                                  df['pickup_pca0'])


def add_distance_features(df):
    lat_km = 111
    lon_km = 85
    df['distance'] = compute_distance_from_lat_long(df['pickup_latitude'], df['pickup_longitude'],
                                                    df['dropoff_latitude'], df['dropoff_longitude'])
    df['manhattan_dist'] = np.abs(df['pickup_latitude'] - df['dropoff_latitude']) * lat_km + np.abs(
        df['pickup_longitude'] - df['dropoff_longitude']) * lon_km


def add_bearing_feature(df):
    df['compass_bearing'] = calculate_bearing(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'],
                                              df['dropoff_longitude'])


def add_time_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],
                                           format='%Y-%m-%d %H:%M:%S',
                                           errors='coerce')
    df['pickup_datetime_day'] = df['pickup_datetime'].dt.day
    df['pickup_datetime_month'] = df['pickup_datetime'].dt.month
    df['pickup_dt_bin'] = ((df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()) // (3 * 3600)
    df['pickup_datetime_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_hour_group'] = df['pickup_datetime'].dt.hour // 4

    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype(int)


def add_log_trip(df):
    df['log_trip_duration'] = np.log1p(df['trip_duration'])


def poly_transform(df):
    poly = PolynomialFeatures(degree=6, include_bias=False, interaction_only=False)
    poly2 = PolynomialFeatures(degree=5, include_bias=False, interaction_only=False)
    poly3 = PolynomialFeatures(degree=7, include_bias=False, interaction_only=True)

    comb1 = ["dropoff_longitude", "dropoff_latitude", "distance", "manhattan_dist", "compass_bearing",
             "vendor_id"]  # deg 6
    comb2 = ["passenger_count", "pickup_datetime_day", "pickup_datetime_month", "pickup_dt_bin",
             "pickup_datetime_dayofweek", "pickup_hour_group", "store_and_fwd_flag"]  # deg 5
    comb3 = ['pickup_pca0', 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1', 'pickup_cluster', 'dropoff_cluster']  # deg 5
    comb4 = ['pickup_lat_bin', 'pickup_long_bin', 'center_lat_bin', 'center_long_bin', 'pca_manhattan']  # deg 5
    comb5 = ['pickup_cluster', 'dropoff_cluster', 'distance', "pickup_datetime_dayofweek", "pickup_hour_group",
             'compass_bearing']  # deg 7, interaction_only=True
    feats = [comb1, comb2, comb3, comb4, comb5]

    X = np.array([])
    for idx, feat_comb in enumerate(feats):
        temp = df[feat_comb]
        temp_np = temp.to_numpy()
        if idx == 0:
            temp_np = poly.fit_transform(temp_np)
        elif idx == 4:
            temp_np = poly3.fit_transform(temp_np)
        else:
            temp_np = poly2.fit_transform(temp_np)

        if idx == 0:
            X = temp_np
        else:
            X = np.hstack((X, temp_np))

    target_column = 'log_trip_duration'
    y = df[target_column].to_numpy()

    return X, y


def transform_train_val_test(X_train, X_val, X_test=None):
    processor = StandardScaler()
    X_train = processor.fit_transform(X_train)
    X_val = processor.transform(X_val)
    if X_test is not None:
        X_test = processor.transform(X_test)
    return X_train, X_val, X_test, processor


def transform_eval(X, processor):
    X = processor.transform(X)
    return X


def log_transform(X_train, X_val = None, X_test=None):
    X_train = np.log1p(np.maximum(X_train, 0))
    if X_val is not None:
        X_val = np.log1p(np.maximum(X_val, 0))
    if X_test is not None:
        X_test = np.log1p(np.maximum(X_test, 0))
    return X_train, X_val, X_test


def model_eval(model, X, y):
    y_pred = model.predict(X)
    print(r2_score(y, y_pred))

if __name__ == '__main__':
    df = pd.read_csv('split/train.csv')
    df_val = pd.read_csv('split/val.csv')

    add_time_features(df)
    add_time_features(df_val)

    add_bearing_feature(df)
    add_bearing_feature(df_val)

    add_distance_features(df)
    add_distance_features(df_val)

    pca, coords = do_pca(df, df_val)

    kmeans = do_kmeans(coords)

    add_kmeans_features(df, kmeans)
    add_kmeans_features(df_val, kmeans)

    add_pca_features(df, pca)
    add_pca_features(df_val, pca)

    add_remove_noise_features(df)
    add_remove_noise_features(df_val)

    add_log_trip(df)
    add_log_trip(df_val)

    X_train, y_train = poly_transform(df)
    X_val, y_val = poly_transform(df_val)

    # print(X_train.shape)

    X_train, X_val, _, processor = transform_train_val_test(X_train, X_val)

    X_train, X_val, _ = log_transform(X_train, X_val)

    model = Ridge(alpha=1)
    model.fit(X_train, y_train)

    # Save the model
    file_name = 'best_model/model.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

    file_name = 'best_model/kmeans.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(kmeans, file)

    file_name = 'best_model/pca.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(pca, file)

    file_name = 'best_model/processor.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(processor, file)

    model_eval(model,X_train,y_train)

    model_eval(model, X_val, y_val)
