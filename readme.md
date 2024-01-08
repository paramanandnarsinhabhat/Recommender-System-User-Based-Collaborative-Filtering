
# User-Based Collaborative Filtering Recommender System

## Introduction
This project implements a User-Based Collaborative Filtering Recommender System using the MovieLens 100K Dataset. The MovieLens 100K dataset is a classic collection of ratings provided by users for movies, and it serves as a benchmark for recommender systems.

## Dataset
The MovieLens 100K dataset includes:
- 100,000 ratings (1-5) from 943 users on 1682 movies.
- Each user has rated at least 20 movies.
- Demographic information for the users (age, gender, occupation, zip code).

The dataset was collected through the MovieLens website from September 19th, 1997 to April 22nd, 1998. For more information and personalized movie recommendations, visit [MovieLens](https://movielens.org/).

## Files
- `data/movie_info.csv`: Contains information about the movies, such as movie ID, title, release date, and genres.
- `data/ratings.csv`: Contains the user ratings for different movies.
- `notebook/user_based_collab_filtering.ipynb`: Jupyter notebook with the analysis and algorithm implementation.
- `scripts/user_based_collaborative_filtering.py`: Python script for the recommender system.
- `requirements.txt`: List of Python packages required for the project.

## Setup
To run this project, ensure that you have the following Python packages installed:
- pandas
- numpy
- scikit-learn

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage
To execute the recommender system, run the Python script located in the `scripts` directory:
```bash
python scripts/user_based_collaborative_filtering.py
```

Alternatively, you can explore the Jupyter notebook in the `notebook` directory for an interactive session.

## Methodology
The recommender system follows these steps:
1. Data Preprocessing: Merge movie information into the ratings dataframe.
2. Data Splitting: Create training and test datasets.
3. Model Building:
   - Implement a simple baseline using the average of all ratings.
   - Build a user-based collaborative filter using mean ratings.
   - Enhance the collaborative filter using similarity-weighted mean ratings.
4. Evaluation: Compute the RMSE (Root Mean Squared Error) to evaluate model performance.

## Contact
For any further questions or contributions, please reach out to the repository owner.


.
