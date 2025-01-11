# NYC_Trip_Duration_Prediction

In this project, I built a model that predicts the total ride duration of taxi
trips in New York City.

This [competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview) was held by Kaggle, the primary dataset is
provided by the NYC Taxi and Limousine Commission which includes
pickup time, geo-coordinates, number of passengers, and several other
variables.

## EDA and Feature Engineering

The key factor in this project is feature engineering that will help you build a good model, also i used unsupervised learning techniques
to invent new features which are kmeans and PCA.

Also i added distance feature, which has a strong correlation with trip duration.

See Taxi_Trip_Duration(EDA).ipynb file for full analysis.

## Data and Modeling Pipeline

<div>
<img src="https://github.com/user-attachments/assets/865083f4-89b5-45e6-8c61-b5de57ceafdc" width = "500">
</div>

In the pipeline I split the features into 4 parts each of them are
transformed with PolynomialFeatures either with degree 5 or 6, then I took
one or two features from each of the 4 parts, and made a combination of
them and transform it with PolynomialFeatures with degree 7, after that I did
a StandardScaler then log transformation.
The final model used is regularized linear regression, Ridge with alpha=1.


## How to configure this project for your own uses

I'd encourage you to clone and rename this project to use for your own puposes.

You will need to install the following libraries:
pandas, scikit-learn, numpy, matplotlib, seaborn
