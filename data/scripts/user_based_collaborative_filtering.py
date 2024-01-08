#User Based Collaborative Filtering
## Movielens - 100K Dataset
'''
MovieLens 100K dataset has been a standard dataset used for benchmarking recommender systems for more than 20 years now and hence this provides a good point to start our learning journey for recommender systems. For non commercial personalised recommendations for movies you can check out the website: https://movielens.org/

This data set consists of:
	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
	* Each user has rated at least 20 movies. 
        * Simple demographic info for the users (age, gender, occupation, zip)

The data was collected through the MovieLens web site (movielens.umn.edu) during the seven-month period from September 19th, 1997 through April 22nd, 1998. This data has been cleaned up - users who had less than 20 ratings or did not have complete demographic information were removed from this data set. 
'''
## Data Description
'''
**Ratings**    -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a comma separated list of 
	         user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   



'''
'''**Movie Information**   -- Information about the items (movies); this is a comma separated
              list of
              movie id | movie title | release date | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
'''

'''**User Demographics**    -- Demographic information about the users; this is a comma
              separated list of
              user id | age | gender | occupation | zip code'''

#Step 1 : Reading Dataset

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split






