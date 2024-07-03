import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity # to find similarity between two vectors of an inner product space. Used to measure similarity in text analysis by finding out if the two vectors are pointing in the same direction 

credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")


movies_df = movies_df.merge(credits_df, on = 'title') # This has merged both datasets on the mentioned column
pd.set_option('display.max_columns', None) # This will print all the columns and rows in the terminal
pd.set_option('display.max_rows', None)
# print(movies_df.head())

# movies_df.info() Provides us with each info of the database
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# print(f"Missing values before: {movies_df.isnull().sum()}") || This shows us the number of missing values in each column

movies_df.dropna(inplace=True) # This will drop entries with missing values

# print(f"Duplicated values before: {movies_df.duplicated().sum()}") || If we remove the .sum(), it will show every row with 'False/True' return with it
movies_df.drop_duplicates()

# print(movies_df.iloc[0].title) || iloc function helps us select any specific row or column. This is showing title of the movie on row 0

def convert(obj): # We have created a function which will take the name from each entry and append it in an array. This is done to make the dataset more easier to read
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L



# print(f"Dataset before: {movies_df.head()}")
movies_df['genres'] = movies_df['genres'].apply(convert) # We have updated the genres category here. Now it contains only the "name" column 
movies_df['keywords'] = movies_df['keywords'].apply(convert)
# print(f"Dataset after: {movies_df.head()}")

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# print(f"Dataset before: {movies_df.head()}")
movies_df['cast'] = movies_df['cast'].apply(convert3)
# print(f"Dataset after: {movies_df.head()}")

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# print(f"Dataset before: {movies_df.head()}")
movies_df['crew'] = movies_df['crew'].apply(fetch_director)
# print(f"Dataset after: {movies_df.head()}")

# print(movies_df['overview'][0]) || This gives us the overview of the first movie entry in the database

# The overview contains long passages which will be difficult to use in the recommendation engine so we will seperate them
# print(f"Dataset before: {movies_df.head()}")
movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split()) # This will split each word in the text 
# print(f"Dataset after: {movies_df.head()}")

# We will now remove the spaces between words if there are any for each category
# print(f"Dataset before: {movies_df.head()}")
movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
# print(f"Dataset after: {movies_df.head()}")

# we will create a new column in which we will add all the data
movies_df['tags'] = movies_df['cast'] + movies_df['crew'] + movies_df['genres'] + movies_df['keywords'] + movies_df['overview']
# print(f"{movies_df.head()}") 

# we will now create a new database 
new_db = movies_df[['movie_id', 'title', 'tags']]

# we will remove the array brackets from the tags column here
new_db['tags'] = new_db['tags'].apply(lambda x: ' '.join(x))

# we will now convert all the data in tags column to lowercase
new_db['tags'] = new_db['tags'].apply(lambda x: x.lower())
# print(new_db.head())


# The data has been vectorized 
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_db['tags']).toarray()
# print(vector.shape) 
# print(vector[0])

# print(len(cv.get_feature_names_out())) || This shows the number of vectors

ps = PorterStemmer() # to reduce a word to its word stem
def stem(text): # This will stem every text
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# print(f"Dataset before: {new_db.head()}")
new_db['tags'] = new_db['tags'].apply(stem)
# print(f"Dataset after: {new_db.head()}")

# print(cosine_similarity(vector)) || This shows the vectors
# print(cosine_similarity(vector).shape)
similarity = cosine_similarity(vector)

# print(similarity[0])

# Now we will create a sorted list
# print(sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1]))[1:6]
# the enumerate function converts a data collection into an enumerate object 
# enumerate returns an object that contains a counter as a key for each value with an object making item within a collection easier to access


# Now we will make the recommendation system
def recommend_movie(movie):
    movie_index = new_db[new_db['title'] == movie].index[0] # we check the movie given and save the movie index
    distances = similarity[movie_index] # we calculate the distance by putting the movie index
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_db.iloc[i[0]].title)
    
print("First movie recommendation:")
print(recommend_movie('Avatar'))
print("Second movie recommendation:")
print(recommend_movie('Iron Man'))
# print(recommend_movie('Spider Man'))
