# Airbnb-Analysis

## INTRODUCTION
**Problem Statement:**

The goal of this analysis is to analyze the factors that determine the price of Airbnb listing in New York City. Using a variety of factors such as neighborhood, burrow, proximity to New York City attractions, number of bedrooms, and type of listing, the aim is to estimate the price of listings.

**Significance of the Problem:**

With a model to estimate the average prices of Airbnbs in the area, Airbnb owners can better understand how to modify the prices off their listings to become competitive in the market. In addition, for those planning to set up an Aribnb, understanding which features are give the most importance can be crucial to the sucees of a listing.

# METHOD
### Data Acquisition

* I utilized a dataset on Kaggle, which has ~49,000 Airbnb listings oin Ney York City in 2019. Each of thie listings came with information such as the  host ID, neighbourhood group, neighbourhood, latitude, longitude, room type, price, minimum nights, and number of reviews. 
Kaggle Dataset: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

* In addition to this information, I wanted to include the distance form each of these listings to the most popular tourist destinations in New York City. In order to do so, I web scraped a tourist site containing this information and used Google's Geolocation API to record the respective latitudes and longitudes. This data was later used to caluculate the distances between each listing and the tourist destinations.

### Variables
* Feature Variables supplied: 
    * listing ID: ID assigned by Airbnb
    * Name of the Lisiting: Description of the Airbnb on the website
    * Host ID: ID of the host assigned by Airbnb
    * Name of the Host: Owner of the listing
    * Location: Manhattan, Brooklyn, Queens, Bronx, Staten Island
    * Neighborhood: the neighbourhood the listing is located in (ex: Harlem, Midtown)
    * Latitude: Coordinate of the listing
    * Longitude: Coordinate of the listing
    * Listing Space Type: Private Home vs Entire home/apt 
    * Distances from each of the most popular tourist sites in NYC
* Target Variable: Price of the listing (in dollars)


### Summary of Analysis

In order to analyze our data, I began by visualizing the dataset through the use of boxplots, histograms, and a series of maps that that plotted the individual tourist sites, listings, and both together on a map on New York City. Then we proceeded to use feature engineering to exclude a few categorical variables and split variables such as Location and Neighborhood into groups. I proceeded to test a few classifiers such as Linear, Lasso, Ridge, kNN, SVR, and the MLP regressors. In order to improve the performance, I ran the regressors once again while using the natural log of the prices in order to have a normal distribution. This greatly improved the accuracy of the Linear, Ridge, SVR, and Lasso regressors. However, the R-squared values were quite low. Based on these results, I used Recursive Feature Elimination in order to reduce the number of variables in the dataset in hopes of improving the accuracy of the kNN regressor. As a result, this modification slightly improved the performance of the kNN regressor. However, the MLP regressor had the better performance when using all 290 features.   


### Interpretation of Findings

***Algorithms Compared***
I compared Linear, Ridge, Lasso, kNN, SVR, and  the MLP regressors.

***Algorithms with Best Performance***
MLP Regression had an R-squared value of 0.4724 on the training set and 0.4218 on the testing set. This indicates that the model fit the dataset quite well and the testing set showed that this model has potential as it has ~65% accuracy.

kNN Regression had an R-squared value of 0.4889 on the training set and 0.3455 on the testing set when using Recursive Feature Elimination. This indicates that the model slightlty overfit the dataset and the testing set showed that this model was an average fit for the dataset, but did not perform as well as the MLP regressor for the given data.

***Algorithms for Use in Predictive Model***

The MLP regressor had the best fit for this dataset and utilized a variety of variables to output better predictions. This regressors also embodies the fact that a large number of features are important when determining where to open an Airbnb and at what price to do so. This model has a ~65% accuracy and that is a reasonable accuracy for a versatile question.
