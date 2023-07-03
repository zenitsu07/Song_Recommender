pip show spotipy
!pip install python-dotenv spotipy

import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

#to slow down code
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)

spotify_df = pd.read_csv(path)
spotify_df.info()
spotify_df.isnull().sum()

corr_matrix = spotify_df.drop(columns=['id','name','release_date','year','artists'])
corr_matrix.corr()

import seaborn as sns

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

#Using minmaxscalar method to normarize the data with columns seelted only of types = int and float variables
from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler -> where the minimum of feature is made equal to zero and the maximum of feature equal to one.

datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = spotify_df.select_dtypes(include=datatypes)
for col in normarization.columns:
MinMaxScaler(col)

normarization

#Use K-means clustering to allot different
from sklearn.cluster import KMeans
#buulding 10 clusters 
kmeans = KMeans(n_clusters=10)
#fitting clusters in normarized data
features = kmeans.fit_predict(normarization)
spotify_df['features'] = features
#There is another way of data scaling, MinMaxScaler -> where the minimum of feature is made equal to zero and the maximum of feature equal to one. 
MinMaxScaler(spotify_df['features'])


#Using the data to build recommender system for given song
class Spotify_Recommendation():
    def __init__(self,dataset):
        self.dataset = dataset
    def recommend_songs(self,song,amount=1):
        #initialiseda as empty array
        distance =[]
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0
            for col in np.arange(len(rec.columns)):
                if not col in [1, 6, 12, 14, 18]:
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]

recommendations = Spotify_Recommendation(data)
recommendations.recommend("Mixe", 10)

