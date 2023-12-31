
# Recommendation Engine for Online Products.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recommendation-system-proj.streamlit.app/)

In this project, we implemented recommendation engines, which are algorithms crafted to analyze user preferences and item characteristics to provide personalized suggestions. We conducted our analysis using an extensive Amazon dataset, which includes ratings for a wide range of electronic products. This dataset is highly detailed, offering valuable insights into user preferences and product feedback.

## Dataset

The dataset is sourced from Kaggle : [Amazon Product Reviews](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)


## EDA and Data Preparation

Exploring the sparsely populated data and preparing it for a recommendation system prediction task.


## Non-Personalized Recommendations 

To find the most popular products in the Amazon dataset, we excluded items with fewer than 50 ratings. This ensured that our results were not skewed by products with limited feedback. We then calculated the average rating for each remaining product and sorted the dataset based on these averages.

This type of recommendations we will generate are called non-personalized recommendations. They are called this as they are made to all users, without taking their preferences into account. This might not select the 'best' items or items that are most suited to the costumer, but there is a good chance they will not hate them as they are so common.


## Rank Based Collaborative Filtering

In this approach, the entire dataset is used to make recommendations. The system analyzes the preferences and behaviors of all users to find similarities between them. To recommend items based on similar users' tastes.



## Model based collaborative  filtering (KNN)

Using the K-Nearest Neighbors (KNN) algorithm as a tool for collaborative filtering we could predict how a user might rate an item even if it is not similar to any item they have seen, by exploiting the similarity between users based on their ratings.

## Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recommendation-system-proj.streamlit.app/)

To enhance interaction with the model, we've developed a user-friendly graphical interface using Streamlit.

This interface consists of three pages:

- The first page of our interface enables users to predict (using KNN) the rating of a specific product by a specific user.
- The second page of our interface enables the user (using Non-Personalized Recommendations) to get top n products based on highest average rating and minimum interactions.
- On The final page of our interface we can look for users who are similar to a specific user then generate recomendations for that user based on the ratings provided by those similar users.
