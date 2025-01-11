import pandas as pd
import pickle
from train import model_eval, add_time_features, add_bearing_feature
from train import add_distance_features, add_kmeans_features, add_pca_features, add_remove_noise_features
from train import add_log_trip, poly_transform, transform_eval, log_transform

def prepare_data(df, pca, kmeans, processor):
    add_time_features(df)

    add_bearing_feature(df)

    add_distance_features(df)

    add_kmeans_features(df, kmeans)

    add_pca_features(df, pca)

    add_remove_noise_features(df)

    add_log_trip(df)

    # print(df.info())

    X_test, y_test = poly_transform(df)

    # print(X_test.shape)

    X_test = transform_eval(X_test, processor)

    X_test, _, _ = log_transform(X_test)

    return X_test, y_test




if __name__ == '__main__':
    file_name = 'best_model/model.pkl'
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    file_name = 'best_model/pca.pkl'
    with open(file_name, 'rb') as file:
        pca = pickle.load(file)

    file_name = 'best_model/kmeans.pkl'
    with open(file_name, 'rb') as file:
        kmeans = pickle.load(file)

    file_name = 'best_model/processor.pkl'
    with open(file_name, 'rb') as file:
        processor = pickle.load(file)

    # df_test = pd.read_csv('split/train.csv')
    df_test = pd.read_csv('split/val.csv')
    # df_test = pd.read_csv('split/test.csv')
    # Uncomment your wanted dataset

    X, y = prepare_data(df_test, pca, kmeans, processor)

    model_eval(model,X,y)


# Train: 0.6779610750919141
# Val: 0.655190462155564
# Test: 0.6785331462450861

