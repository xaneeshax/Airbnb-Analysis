# In[1]:
# Finding New York's Most Popular Tourist Sites: Web Scraping

import pandas as pd
import urllib.request as urllib
from bs4 import BeautifulSoup

# In[2]:
# Reads the URL

URL = 'https://www.thecrazytourist.com/top-25-things-to-do-in-new-york-city/'
html = urllib.urlopen(URL)
print(html.read())


# In[3]:
# Creates a Soup Object

html = urllib.urlopen(URL)
soup = BeautifulSoup(html.read())


# In[4]:
# Compile a list of all the headers

places = soup.find_all('h2')


# In[5]:
# Places in the list

places[::5]


# In[6]:
# Strip headers that do not refer to tourist sites

places = places[1:-3]


# In[7]:
# Strip the strings of unecessary characters

for i in range(len(places)):
    places[i] = str(places[i])
    space = places[i].find(' ')
    places[i] = places[i][(space + 1) :]
    delimeter = places[i].find('</')
    places[i] = places[i][:delimeter]


# In[8]:
# List of Tourist Sites

places[::5]


# In[9]:
# ### Finding the Coordinates of Popular Tourist Sites: Google Geolocation API

import requests
key = 'AIzaSyALsS-**********************'
geocode_url = 'https://maps.googleapis.com/maps/api/geocode/json?'


# In[10]:
import pandas as pd
pd.set_option('display.max_columns', None)

# Use list of locations to create a dataframe
destinations = pd.DataFrame({'places' : places})
destinations.head()


# In[11]:
# Getting latitude and longitude information

latitudes = []
longitudes = []

for place in places:
    
    # Parameters for API request
    parameters = {'key' : key,
                  'address' : place + ' New York City'}
    
    # Get and format response
    response = requests.get(geocode_url, parameters)
    response = response.json()

    # Parse data to find the latitiude and longitude
    coordinates = response['results'][0]['geometry']['location']
    latitudes.append(coordinates['lat'])
    longitudes.append(coordinates['lng'])


# In[12]:
# Input latitude and longitude data into the dataframe

destinations['latitude'] = latitudes
destinations['longitude'] = longitudes
destinations['type'] = ['tourist'] * destinations.shape[0]
destinations.head()

# Convert into a CSV to avoid additional API requests
destinations.to_csv('destinations.csv')


# In[13]:
# Read the csv file

destinations = pd.read_csv('destinations.csv')
destinations = destinations.drop('Unnamed: 0', axis = 1)
destinations.head()


# In[14]:
# Plotting the Tourist Sites and Airbnb Listings: Plotly & Mapbox

import plotly.express as px

# Mapbox Token
token = 'pk*************************'

fig = px.scatter_mapbox(destinations, lat = "latitude", lon = "longitude", 
                        hover_name = "places", color = 'type', 
                        color_discrete_sequence = ["#000000"], size_max = 15, 
                        zoom = 10, height = 400)

fig.update_layout(mapbox_style = "light", mapbox_accesstoken = token)
fig.update_layout(margin={"r" : 0.5, "t" : 0.5, "l" : 0.5, "b" : 0.5})
fig.show()


# In[15]:
# NYC Airbnbs Information

df = pd.read_csv('airbnb_nyc.csv')
df.head()


# In[16]:
# Quarter Percent Trimmed Data

df.shape[0] / 400


# In[17]:
# Sort teh dataframe by the price column

df['price'].values.sort()


# In[18]:
# Get the values of the column

pl = df['price'].values
print(pl[122], pl[-122])



# In[19]:
# Dataset after the quarter percent trimmed data

df = df[df['price'] >=24]
df = df[df['price'] <= 1763]


# In[20]:
# Reset the index

df = df.reset_index()
df = df.drop('index', axis = 1)
df.head()


# In[21]:
# Histogram of Airbnb Prices

fig = px.histogram(df, x = 'price', labels = {'price' : 'Price'},
                   title = 'Distribution of Airbnb Prices in NYC', nbins = 100)
fig.show()


# In[22]:
# Boxplot representing teh prices in each burrow of NYC

fig = px.box(df, x = "neighbourhood_group", y = "price",
             labels = {'neighbourhood_group' : 'New York Burrow', 
                       'price' : 'Price'},
             title = 'Distribution of Airbnb Prices in New York Burrows', 
             color = 'neighbourhood_group')
fig.show()


# In[23]:
#Describes the cost for each location and room type
print(df.groupby(['neighbourhood_group', 'room_type'])['price'].describe())


# In[24]:


# All Airbnb Listings Plotted
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="name", 
                        hover_data=["neighbourhood", "price"],
                        color = 'neighbourhood_group', size = 'price', 
                        size_max = 10,
                        color_discrete_sequence = ["#F6BF26", "#33b7b7", 
                                                   "#c0198b", "#7CB342", 
                                                   "#EF553B"],
                        zoom = 10, height = 400)

fig.update_layout(mapbox_style="light", mapbox_accesstoken=token)
fig.update_layout(margin={"r" : 0.5, "t" : 0.5, "l" : 0.5, "b" : 0.5})
fig.show()




# In[25]:
# Merging the Airbnb dataframe and Destinations dataframe in order to plot 
# both on one map

plotting = df[['name', 'latitude', 'longitude', 'neighbourhood_group', 'price']]
plotting.set_index('name')


# In[26]:
# Set index to name

destinations2 = destinations.rename(columns = {'places' : 'name'})
destinations2 = destinations2.set_index('name')
destinations2.head()


# In[27]:
# Outer merge on the name column

nyc = pd.merge(plotting, destinations2, on = 'name', how = 'outer')
nyc["Size"] = nyc['price']


# In[28]:
# Input NaN values

default = {}
for header in list(nyc.columns):
    if header == 'latitude_x' or header == 'longitude_x':
        default[header] = -1
    elif header == 'neighbourhood_group':
        default[header] = 'Tourist Attraction'
    elif header == 'type':
        default[header] = 'airbnb'
    elif header == 'Size':
        default[header] = 200


# In[29]:
# Fill NaN values

nyc = nyc.fillna(default)
nyc


# In[30]:
# Reconfigure the dataset

for col in [1, 2, -1, -3]:
    for index in range(nyc.shape[0]):
        if nyc.iloc[index, col] == -1 and col == 1:
            nyc.iloc[index, col] = nyc.iloc[index, -4]
        elif nyc.iloc[index, col] == -1 and col == 2:
            nyc.iloc[index, col] = nyc.iloc[index, -3]


# In[31]:
# Dropping extra columns

nyc = nyc.drop(['latitude_y', 'longitude_y'], axis = 1)
nyc


# In[32]:
# Rename columns

nyc = nyc.rename(columns = {'name' : 'Name', 'latitude_x' : 'Latitude', 
                            'longitude_x' : 'Longitude',
                            'neighbourhood_group' : 'Neighbourhood', 
                            'price' : 'Price', 'type' : 'Type'})
nyc


# In[33]:
# Capitalize columns values

nyc['Type'] = nyc['Type'].str.title()


# In[34]:
# Map of Tourist Sites and Airbnbs

fig = px.scatter_mapbox(nyc, lat="Latitude", lon="Longitude", hover_name="Name",
                        hover_data=["Neighbourhood", 'Type', "Price", 
                                    'Latitude', 'Longitude'],
                        color = 'Neighbourhood', size_max = 10, size = 'Size',
                        color_discrete_sequence = ["#F6BF26", "#33b7b7", 
                                                   "#c0198b", "#7CB342", 
                                                   "#EF553B", "#000000"],
                        zoom = 10, height = 400)

fig.update_layout(mapbox_style="light", mapbox_accesstoken=token)
fig.update_layout(margin={"r" : 0.5, "t" : 0.5, "l" : 0.5, "b" : 0.5})
fig.show()


# In[35]:
# Feature Engineering with One-Hot Encoder

from sklearn.preprocessing import OneHotEncoder

def encode_values(values, df):
    
    for value in values:
        
        matrix = df[value].values.reshape(-1, 1)
    
        encoder = OneHotEncoder(sparse = False)
        encoded = encoder.fit_transform(matrix)
        column_df = pd.DataFrame(encoded, columns = encoder.get_feature_names())

        for name in encoder.get_feature_names():
            new_name = name[3:]
            df[new_name] = column_df[name]

        df = df.drop(value, axis = 1)
    
    return df    


# In[36]:
# Encode the values of the main dataframe

encoded_df = encode_values(['neighbourhood_group', 'neighbourhood', 
                            'room_type'], df)
encoded_df.head()


# In[37]:
# Drop extra columns

encoded_df = encoded_df.drop(['id', 'name', 'host_id', 'host_name', 
                              'last_review', 'reviews_per_month'], axis = 1)
encoded_df.head()


# In[38]:
# Distances between Airbnb Listings and Tourist Sites Method


def manhattan_distance(df, destinations):
    
    # For Destinations Dataframe
    NAME = 0
    LAT = 1
    LON = 2
    
    # For df
    LATITUDE = 0
    LONGITUDE = 1
    
    for dest_ind in destinations.index:
        distances = []
        name = destinations.iloc[dest_ind, NAME]
        
        # Calculates each distance and adds them
        for abnb_ind in df.index:
            x = abs(df.iloc[abnb_ind, LATITUDE] - destinations.iloc[dest_ind, LAT])
            y = abs(df.iloc[abnb_ind, LONGITUDE] - destinations.iloc[dest_ind, LON])
            distances.append(x + y)
           
        # Creates column for the dataframe
        df['Dist from ' + name] = distances
        
    return df


# In[39]:
# Updates the dataframe to contain how far each location is from each of these tourist sites
encoded_df = manhattan_distance(encoded_df, destinations)


# In[40]:
# View the data after using one hot encoder

encoded_df


# In[41]:
# Regression Models
# TARGET VARIABLE: PRICE

target = encoded_df['price']
target


# In[42]:
# FEATURES

encoded_df = encoded_df.drop(['price'], axis = 1)
features = encoded_df


# In[43]:
# FUnction to view the coeffient assciated with each variable

def model_coefficeints(features, model):
    """weight for each variable in regression formula"""
    print('\nREGRESSION COEFFICIENTS:')
    for i, name in enumerate(features.columns):
        print(f'\t{name : >10}: {model.coef_[i]}')


# In[44]:
# Show predicted vs estimated values

def model_comparision(predicted, expected):
    """print Some predicted and estimated values"""
    print('\nPREDICTED & ESTIMATED VALUES:')

    for p, e in zip(predicted[:5], expected[:5]):
        print(f'\tpredicted: {p:.2f}, expected: {e:.2f}')


# In[45]:
# Creating the regression model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

regressors = {'LINEAR REGRESSION': LinearRegression(), 
              'RIDGE REGRESSION' : Ridge(alpha = 100),
              'LASSO REGRESSION' : Lasso(alpha = 0.001),
              'K-NEIGHBORS REGRESSION' : KNeighborsRegressor(n_neighbors = 8),
              'SVR REGRESSION' : LinearSVR(C=0.01, max_iter = 10000000),
              'MLP REGRESSION' : MLPRegressor(max_iter = 6000)
             }

def regression_analysis():
    # split data into training and testing
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                        random_state = 3000)

    for regressor_item, regressor_object in regressors.items():

        print('\n' + regressor_item + '\n------------------------\n')
        
        # Train data using each regressor
        model = regressor_object.fit(X = X_train, y = y_train)
        predicted = model.predict(X_test)
        expected = y_test

        # Show metrics
        print('\tR-squared value for training set: ', 
              r2_score(y_train, model.predict(X_train)))
        
        print('\tR-squared value for testing set: ', 
              r2_score(y_test, model.predict(X_test)))
        
        print('\n\tAverage Deviation from Expected Value: ', 
              mean_squared_error(expected, predicted, squared = False))

            


# In[46]:
# Run the regression function
regression_analysis()


# In[47]:
# Logarithmic Engineering the Target Variables
# Since the target variable has a heavy skew -> 
# normalize the dataset by taking the log of each price
import numpy as np

orig_target = target.values
log_target = np.log(target.values)
log_target


# In[48]:
# Histogram of Airbnb Prices

fig = px.histogram(x = log_target, labels = {'x' : 'Log Price'}, 
                   title = 'Distribution of Airbnb Prices in NYC', nbins = 40)
fig.show()


# In[49]:
# Update the target value
target = log_target


# In[50]:
# Perform regression analysis with the updated data

regression_analysis()

# ### Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

# Uses a Decision Tree Regressor to eliminte features that have little to no 
# effect on the target variable
select = RFE(DecisionTreeRegressor(random_state = 3000), n_features_to_select = 50)

rfe_columns = []
for i in range(len(select.get_support())):
    if select.get_support()[i] == True:
        rfe_columns.append(features.columns[i])

small_df = df[rfe_columns]

# In[51]:
# Used to store the column names of the Top 50 Features

pd.DataFrame({'names' : small_df.columns}).to_csv('FEATURES50.csv')


# In[52]:
# Use a dataframe that holds the 50 most impactul variables

reduced = pd.read_csv('FEATURES50.csv')
reduced = list(reduced['names'])
reduced[::5]


# In[53]:
# Use the specified subset of the features

features = encoded_df[reduced]
features


# In[54]:
# Run regression analysis on the Top 50 feautures

target = orig_target
regression_analysis()

