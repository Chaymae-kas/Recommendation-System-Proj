import numpy as np
import pandas as  pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import zipfile
import os

import os
import zipfile
import pandas as pd

def unzip_csv_files(zip_path, extracted_dir):
    try:
        # Ensure the extraction directory exists
        os.makedirs(extracted_dir, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

        # Check if the CSV files exist in the extraction directory
        matrix_filled_path = os.path.join(extracted_dir, 'matrix_filled.csv')
        matrix_w_NANs_path = os.path.join(extracted_dir, 'matrix_w_NANs.csv')

        if os.path.isfile(matrix_filled_path) and os.path.isfile(matrix_w_NANs_path):
            # Read the CSV files
            matrix_filled = pd.read_csv(matrix_filled_path)
            matrix_w_NANs = pd.read_csv(matrix_w_NANs_path)

            # Return the dataframes
            return matrix_filled, matrix_w_NANs
        else:
            print("CSV files not found in the extraction directory.")
            return None, None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

# Specify the paths for the ZIP file and extraction directory
zip_path = 'data.zip'
extracted_dir = 'extracted_file/'

# Call the function to unzip and read the CSV files
matrix_filled, matrix_w_NANs = unzip_csv_files(zip_path, extracted_dir)

# Check if the dataframes were successfully loaded
if matrix_filled is not None and matrix_w_NANs is not None:
    # Perform further processing or analysis with the dataframes
    print("CSV files successfully loaded.")
    # ... Your code here ...
else:
    print("Unable to load CSV files.")

def predict_rating(target_product, target_user, matrix_w_NANs, matrix_filled):
    
    matrix_filled.drop(target_product, axis=1, inplace=True)
    target_user_x = matrix_filled.loc[[target_user]]
    
    other_users_y = matrix_w_NANs[target_product]
    other_users_x = matrix_filled[other_users_y.notnull()]
    
    other_users_y.dropna(inplace=True)  
    
    
    user_knn = KNeighborsRegressor(metric='cosine', n_neighbors=3) 
    user_knn.fit(other_users_x, other_users_y)
    
    user_user_pred = user_knn.predict(target_user_x)
    
    return user_user_pred


def similar_users(user_index, interactions_matrix):
    similarity = []
    for index, row in interactions_matrix.iterrows():

        # finding cosine similarity between the user_id and each user
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [row])

        # Appending the user and the corresponding similarity score with user_id as a tuple
        similarity.append((index, sim))

    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [Tuple[0] for Tuple in similarity]  # Extract the user from each tuple in the sorted list
    similarity_score = [Tuple[1] for Tuple in similarity]  # Extracting the similarity score from each tuple in the sorted list

    # Remove the original user and its similarity score and keep only other similar users
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])

    return most_similar_users

def recommendations(user_index, num_of_products, interactions_matrix):
    # Saving similar users using the function similar_users defined above
    most_similar_users = similar_users(user_index, interactions_matrix)[0:10]

    # Finding product IDs with which the user_id has interacted
    prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))
    recommendations = []

    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            # Finding 'n' products which have been rated by similar users but not by the user_id
            similar_user_prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break

    return recommendations[:num_of_products]

def top_n_products( n, min_interaction):
    #Calculate the average rating for each product 
    average_rating = df_final.groupby(['prod_id']).mean().rating
    #Calculate the count of ratings for each product
    count_rating = df_final.groupby(['prod_id']).count().rating

    #Create a dataframe with calculated average and count of ratings
    final_rating = pd.DataFrame(pd.concat([average_rating,count_rating], axis = 1))
    final_rating.columns=["Average Rating", "Ratings Count"]

    #Sort the dataframe by average of ratings
    final_rating = final_rating.sort_values(by='Average Rating', ascending=True)

    finaal_rating = final_rating.sort_values(by='Ratings Count', ascending=True)
    
    recommendations = finaal_rating[finaal_rating['Ratings Count'] >= min_interaction]
    
    return recommendations.index[:n]

def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
         f"""
        <style>
        .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
         }}
        </style>
        """,
        unsafe_allow_html=True
         )
add_bg_from_local('C:/Users/dell/OneDrive/Bureau/git/Recommendation-System-Proj/background.avif')

with st.sidebar:
   
    selected = option_menu('Recommendation System For Online Products',
                          
                          ['Rating Prediction',
                           'Rank Based Recommendation',
                           'Collaborative Filtering based Recommendation'],
                          icons=['star-half','list-stars','heart-half'],
                          default_index=0)
    



if (selected == 'Rating Prediction'):
    
    st.title('Rating Prediction using KNN')
    
    target_product = st.text_input("Enter the target product ID:")
    target_user = st.text_input("Enter the target user ID:")
    
    prediction = ''
    
    if st.button('Predict score'):
        prediction = predict_rating(target_product, target_user, matrix_w_NANs, matrix_filled)
        single_prediction = prediction[0]  # Convert the NumPy array to a single float
        formatted_prediction = round(single_prediction, 1)
        st.success(formatted_prediction)

    


if (selected == 'Rank Based Recommendation'):

    st.title('Rank Based Recommendation System')

    n=st.text_input("Number of products to recommend :")
    min_interaction=st.text_input("minimum of rating counts :")
    
    topProduct=[]

    if st.button('Top products'):
        topProduct = top_n_products(int(n),int(min_interaction))
        
    st.success(list(topProduct))
    

if (selected == 'Collaborative Filtering Based Recommendation'):

    st.title('Collaborative Filtering Based Recommendation System')
    
    user_index = st.text_input("Enter the user ID:")
    
    most_similar_users =[]
    recommended_products=[]
    
    if st.button('Find similar users'):
        most_similar_users = similar_users(user_index, matrix_filled)[0:10]
    
    st.success(most_similar_users)
    
    num_of_products = st.text_input("Enter the number of products to recommend:")
        
    if st.button('recommend products'):
        recommended_products = recommendations(user_index, int(num_of_products), matrix_filled)
    
    st.success(recommended_products)
    
