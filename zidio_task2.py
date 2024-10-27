#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset (assuming it's in CSV format)
df = pd.read_csv('D:/Datasets/netflix_data.csv')

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())


# In[2]:


# Fill missing values in 'director', 'cast', and 'country' with 'Unknown'
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Check for missing values after preprocessing
print(df.isnull().sum())


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Combine 'listed_in' and 'description' columns to build content features
df['content'] = df['listed_in'] + " " + df['description']

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the content into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['content'])

# Check the shape of the TF-IDF matrix
print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}')


# In[4]:


from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity between all shows
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Check the shape of the similarity matrix
print(f'Cosine Similarity Matrix Shape: {cosine_sim.shape}')


# In[5]:


# Create a series with indices for reverse lookups
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Function to get recommendations based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get pairwise similarity scores of all shows with that show
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort shows by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar shows
    sim_scores = sim_scores[1:6]

    # Get the indices of those shows
    show_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar shows
    return df['title'].iloc[show_indices]

# Example: Get recommendations for a specific title
print(get_recommendations('Kota Factory'))


# In[6]:


import numpy as np

# Create a pivot table with random user interactions (this simulates user behavior)
np.random.seed(0)
df['user_id'] = np.random.randint(1, 1000, df.shape[0])

user_item_matrix = df.pivot_table(index='user_id', columns='title', aggfunc='size', fill_value=0)

# Check the user-item matrix
print(user_item_matrix.head())


# In[7]:


from sklearn.neighbors import NearestNeighbors

# Instantiate the NearestNeighbors model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit the model on the user-item matrix
model_knn.fit(user_item_matrix)

# Example: Get 5 nearest neighbors for a given user (user_id=1)
distances, indices = model_knn.kneighbors(user_item_matrix.loc[1].values.reshape(1, -1), n_neighbors=5)

# Display nearest users
print(f'Nearest Users for User 1: {indices.flatten()}')


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap for the cosine similarity matrix (for first 10 shows)
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim[:10, :10], annot=True, cmap='coolwarm', xticklabels=df['title'][:10], yticklabels=df['title'][:10])
plt.title("Cosine Similarity Matrix (First 10 Shows)")
plt.show()


# In[9]:


# Plot the distribution of content ratings
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='rating', order=df['rating'].value_counts().index, palette='Set3')
plt.title('Distribution of Content Ratings', fontsize=14)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.show()


# In[10]:


# Split the 'listed_in' column into individual genres
df['genres'] = df['listed_in'].str.split(',')

# Explode the 'genres' column to count each genre separately
df_genres = df.explode('genres')

# Count the occurrences of each genre
top_genres = df_genres['genres'].value_counts().nlargest(10)

# Bar plot of top 10 genres
plt.figure(figsize=(10, 6))
sns.barplot(y=top_genres.index, x=top_genres.values, palette='viridis')
plt.title('Top 10 Genres by Content Count', fontsize=14)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.show()


# In[12]:


# EDA Plot 1: Distribution of Content Type (Movies vs TV Shows)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='type', palette='Set2')
plt.title('Distribution of Content Type (Movies vs TV Shows)', fontsize=14)
plt.xlabel('Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# In[13]:


# EDA Plot 2: Top 10 Countries by Content Production
top_countries = df['country'].value_counts().nlargest(10)

plt.figure(figsize=(6, 6))
sns.barplot(y=top_countries.index, x=top_countries.values, palette='coolwarm')
plt.title('Top 10 Countries by Content Production', fontsize=14)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()


# In[ ]:




