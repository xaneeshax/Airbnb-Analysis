# Airbnb-Analysis

## INTRODUCTION
**Problem Statement:**

The goal of this analysis is to analyze the factors that determine the price of Airbnb listing in New York City. Using a variety of factors such as neighbourhood, burrow, proximity to New York City attractions, number of bedrooms, and type of listing, the aim is to estimate the price of listings.

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
    * Neighbourhood: the neighbourhood the listing is located in (ex: Harlem, Midtown)
    * Latitude: Coordinate of the listing
    * Longitude: Coordinate of the listing
    * Listing Space Type: Private Home vs Entire home/apt 
    * Distances from each of the most popular tourist sites in NYC
* Target Variable: Price of the listing (in dollars)

