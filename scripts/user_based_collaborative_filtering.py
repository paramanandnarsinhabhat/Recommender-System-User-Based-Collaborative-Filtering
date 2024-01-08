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

#Reading ratings file:
ratings = pd.read_csv('/Users/paramanandbhat/Downloads/UserBasedCollaborativeFilteringfromscratch-201024-234223/ratings.csv')

#Reading Movie Info File
movie_info = pd.read_csv('/Users/paramanandbhat/Downloads/UserBasedCollaborativeFilteringfromscratch-201024-234223/movie_info.csv')




#Step 2 : Merging Movie information to ratings dataframe 

'''The movie names are contained in a separate file. Let's merge that data with ratings and store it in ratings dataframe. The idea is to bring movie title information in ratings dataframe as it would be useful later on'''

#Running into an error that movie_is is not found
#Debuging the error


#ratings = ratings.merge(movie_info[['movie_id','movie title']], how='left', left_on = 'movie_id', right_on = 'movie_id')

print(ratings.columns)
print(movie_info.columns)

ratings = ratings.merge(movie_info[['movie id', 'movie title']], how='left', left_on='movie_id', right_on='movie id')


print(ratings.head())

'''Lets also combine movie id and movie title separated by ': ' and store it in a new column named movie'''
ratings['movie'] = ratings['movie_id'].map(str) + str(': ') + ratings['movie title'].map(str)

print(ratings.columns)

#Keep only the required columns and drop rest
ratings = ratings.drop(['movie id', 'movie title', 'movie_id','unix_timestamp'], axis = 1)

ratings = ratings[['user_id','movie','rating']]

print(ratings.columns)

## 3. Creating Train & Test Data & Setting Evaluation Metric

#Assign X as the original ratings dataframe
X = ratings.copy()

#Split into training and test datasets
X_train, X_test = train_test_split(X, test_size = 0.25, random_state=42)

#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

## 4. Simple Baseline using average of all ratings 
#Define the baseline model to always return average of all available ratings
def baseline(user_id, movie):
    return X_train['rating'].mean()

#Function to compute the RMSE score obtained on the test set by a model
def rmse_score(model):
    
    #Construct a list of user-movie tuples from the test dataset
    id_pairs = zip(X_test['user_id'], X_test['movie'])
    
    #Predict the rating for every user-movie tuple
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    
    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    
    #Return the final RMSE score
    return rmse(y_true, y_pred)

rmse_score(baseline)

print(rmse_score(baseline))

#6. User based Collaborative filtering with simple user mean
#Build the ratings matrix using pivot_table function
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie')

r_matrix.head()

print(r_matrix.head())

#User Based Collaborative Filter using Mean Ratings
def cf_user_mean(user_id, movie):
    
    #Check if movie exists in r_matrix
    if movie in r_matrix:
        
        #Compute the mean of all the ratings given to the movie
        mean_rating = r_matrix[movie].mean()
    
    else:
        #Default to average rating from the train set
        mean_rating = X_train['rating'].mean()
    
    return mean_rating

#Compute RMSE for the Mean model
rmse_score(cf_user_mean)

print(rmse_score(cf_user_mean))

## 7. User based Collaborative filtering with similarity weighted mean
#Compute the Pearson Correlation using the ratings matrix with corr function from Pandas
pearson_corr = r_matrix.T.corr()

#Convert into pandas dataframe 
pearson_corr = pd.DataFrame(pearson_corr, index=r_matrix.index, columns=r_matrix.index)

pearson_corr.head(10)

print(pearson_corr.head(10))

#Fill all the missing correlations with 0
pearson_cor = pearson_corr.fillna(0)

'''
Now, we have the user user similarities stored in the matrix pearson_cor. We will define a function to predict the unknown ratings in the test set using user based collarborative filtering with simiarity as pearson correlation and using all neighbours with positive correlation. For each user movie pair:
1. Check if a movie is there in train set, if its not in that case we will just predict the mean rating as the predicted rating
2. Calculate the mean rating for the active user
3. Extract correlation values from matrix pearson_corr and sort it in decreasing order of correlation values
4. Keep only similarity scores for users with positive correlation with the active user
5. Drop all the users similar to active user but haven't rated the target movie
6. Do a check and predict mean rating if there are no similar users who have rated the target movie
'''
#User Based Collaborative Filter using Weighted Mean Ratings
def cf_user_wmean(user_id, movie_id):
    
    #Check if movie_id exists in r_matrix
    if movie_id in r_matrix:
        
        #Mean rating for active user
        ra = r_matrix.loc[user_id].mean()

        #Get the similarity scores for the user in question with every other user
        sim_scores = pearson_corr[user_id].sort_values(ascending = False)
        
        # Keep similarity scores for users with positive correlation with active user
        sim_scores_pos = sim_scores[sim_scores > 0]
        
        #Get the user ratings for the movie in question
        m_ratings = r_matrix[movie_id][sim_scores_pos.index]
        
        #Extract the indices containing NaN in the m_ratings series (Users who have not rated the target movie)
        idx = m_ratings[m_ratings.isnull()].index
        
        #Drop the NaN values from the m_ratings Series
        m_ratings = m_ratings.dropna()
        
        # If there are no ratings from similar users we cannot use this method so we predict just 
        # the average rating of the movie else we use the prediction formula
        if len(m_ratings) == 0:
            #Default to average rating in the absence of ratings by similar users
            wmean_rating = r_matrix[movie_id].mean()
        else:   
            #Drop the corresponding correlation scores from the sim_scores series
            sim_scores_pos = sim_scores_pos.drop(idx)
            
            #Subtract average rating of each user from the rating (rbp - mean(rb))
            m_ratings = m_ratings - r_matrix.loc[m_ratings.index].mean(axis = 1)
            
            #Compute the final weighted mean using np.dot which is nothing but the product divided by sum of weights
            wmean_rating = ra + (np.dot(sim_scores_pos, m_ratings)/ sim_scores_pos.sum())
   
    else:
        #Default to average rating in the absence of any information on the movie in train set
        wmean_rating = X_train['rating'].mean()
    
    return wmean_rating


rmse_score(cf_user_wmean)

print(rmse_score(cf_user_wmean))














