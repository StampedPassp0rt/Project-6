import os
import subprocess
import collections
import re
import csv
import json

import pandas as pd
import numpy as np
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import psycopg2
import requests
from imdbpie import Imdb
import nltk

import urllib
from bs4 import BeautifulSoup
import nltk

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#1. Connect to the IMDB API

imdb = Imdb()
imdb = Imdb(anonymize=True)

#2. Query the top 250 rated movies in the database
top_250 = imdb.top_250()
top_250
#list of dictionaries...

'''#3 3. Make a dataframe from the movie data
Keep the fields:
num_votes
rating
tconst
title
year
And discard the rest'''


top250 = pd.DataFrame(top_250, columns = ['num_votes', 'rating', 'tconst', 'title', 'year'], index = range(0,250))
top250.head()

#Ordered by rating...

#3. Select only the top 100 movies

top100 = top250.sort_values('rating', ascending = False)[0:100]
top100.tail()

'''4. Get the genres and runtime for each movie and add them to the dataframe
There can be multiple genres per movie, so this will need some finessing.'''

#I wonder if the certification is significant too (i.e. PG-13...)
#genre - probably going to end up making genre a series of dummies so that each movie is still just one row...

#Pulling runtime and classification to add to DF

#loop through the DF by id, and for each id, pull the rating and add it...

#Runtime lambda fxn
runtime_pull = lambda x: imdb.get_title_by_id(x).runtime



#Top 100 runtime
top100['runtime'] = top100.tconst.apply(runtime_pull)

#Top 250's runtime
top250['runtime'] = top250[u'tconst'].apply(runtime_pull)


#Genres lambda function to get genres from imdb database
genres_pull = lambda x: imdb.get_title_by_id(x).genres

#Getting genres for top100 DF
top100['genre'] = top100.tconst.apply(genres_pull)

#Getting genres for top250 DF
top250['genre'] = top250[u'tconst'].apply(genres_pull)
print top100.head()

print top250.info()
top250.head()

#Classification lambda fxn
classification_pull = lambda x: imdb.get_title_by_id(x).certification

#Top 100 classification

top100['classification'] = top100.tconst.apply(classification_pull)

#Top 250 classification

top250['classification'] = top250.tconst.apply(classification_pull)

#Getting top100 genres into new dataframe
df_new = pd.DataFrame(top100[['tconst', 'genre']])

#iterating to get genres into a long list
test = []
for i in df_new.index:
    for e in range(len(df_new.genre[i])):
        test.append((df_new.tconst[i], df_new.genre[i][e]))

test
#Putting the list into DF format
top100genres = pd.DataFrame(test, columns = ['tconst', 'genre'])

top100genres_new = top100genres.groupby('tconst').genre.value_counts().unstack()
top100genres_new.fillna(value = 0, inplace = True)
top100genres_new

#one way to get the genre data....

#So I had two ways ultimately. One was where I had multiple rows per movie, essentially where the genres were melted into one column.
#The other way was to essentially treat the genres like dummy variables. The latter feels like the better construction for any regression.
#So despite it being 'long' data, keeping the records to one row per movie, and then a long series of columns of indicator genres for each record probably is better
top100_new = pd.merge(top100, top100genres_new, left_on = 'tconst', right_on = 'tconst', how = 'left', left_index = False, right_index = True)


#Top 250 genres...
#getting the list of genres... probably an easier way to do this...

genre_tconst_list = []
for i in top250.index:
    for e in range(len(top250.genre[i])):
        genre_tconst_list.append((top250.tconst[i], top250.genre[i][e]))

top250genres = pd.DataFrame(genre_tconst_list, columns = ['tconst', 'genre'])
top250genres_new = top100genres.groupby('tconst').genre.value_counts().unstack()
top250genres_new.fillna(value = 0, inplace = True)
top250_new = pd.merge(top250, top250genres_new, left_on = 'tconst', right_on = 'tconst', how = 'left', left_index = False, right_index = True)
top250_new.head()

#4. Write the Results to a csv

top250_new.to_csv('top250_summary.csv', sep = ',', encoding = 'utf-8')

top100_new.to_csv('top100_summary.csv', sep = ',', encoding = 'utf-8')

'''Part 2: Wrangle the text data
1. Scrape the reviews for the top 100 movies

Hint: Use a loop to scrape each page at once

I switched to using API for the review data...
The reason is that I could not get the logic to work for properly pairing header
 and rating (if there) to review text. I know there is a way, but right now it
 is taking too long.

When I had webscraped, I had, for the first 30 movies, 30005 headers and 30001 reviews.
This mismatch indicates my logic was not ideal, especially considering that the API
has a match between the number of review headers (summary) and reviews. My Jupyter notebook can be referenced.

Anyway, something I want to come back to because it obviously is a solvable problem.

 '''

 #Load data from csvs.

top100_new = pd.read_csv('top100_summary.csv')
top250_new = pd.read_csv('top250_summary.csv')
top100_new.drop('Unnamed: 0', axis = 1, inplace = True)
top250_new.drop('Unnamed: 0', axis = 1, inplace = True)

#Pulling all reviews from IMDB with API

review_list = []

for title in top100_new.tconst:
    review = imdb.get_title_reviews(title, max_results = 6000)
    for i in range(len(review)):
        review_list.append((title, review[i].rating,
                            review[i].summary,
                            review[i].text))

review_list[0]

#Checking length of reviews_list

len(review_list) #makes sense

#Putting in DataFrame.

reviews = pd.DataFrame(review_list, columns = ['title', 'rating', 'header', 'text'])
reviews.info() #makes sense. Not all reviews have ratings filled out.

#Stripped out the non-alphanumeric characters
replace = lambda x: re.sub(r'\W+', ' ', re.sub(r'\'', '', x))

reviews['text'] = reviews['text'].apply(replace)

#verifying..
reviews['text'][0]

#saving to csv right now....
reviews.to_csv('reviews_full.csv', sep = ',', index_label = 'tconst', encoding = 'utf-8')

#Loading reviews from csv.


reviews = pd.read_csv('reviews_full.csv')

reviews.head()

#drop first col of csv.
reviews.drop('tconst', axis = 1, inplace = True)
reviews.head()

#2. Extract the reviews and the rating per review for each movie
#Note: "soup" from BeautifulSoup is the html returned from all 25 pages. You'll need to either address each page individually or break them down by elements.


#Done above in API.

#3. Remove the non AlphaNumeric characters from reviews​

#Done above
reviews.head() #reminding myself of the columns.


#4. Calculate the top 200 ngrams from the user reviews
'''Use the TfidfVectorizer in sklearn.
Recommended parameters:
ngram_range = (1, 2)
stop_words = 'english'
binary = False
max_features = 200'''

from sklearn.feature_extraction.text import TfidfVectorizer

#setting up TF-IDF
tfidf = TfidfVectorizer(encoding = 'utf-8', ngram_range = (1,2), strip_accents = 'unicode',
                       stop_words = 'english', binary = False, max_features = 200)

#Fit and Transform our features.

#Review Text only.
textonly_tfidf_dtm = tfidf.fit_transform(reviews.text)

print tfidf.get_feature_names()

dense_textonly = pd.DataFrame(textonly_tfidf_dtm.todense(), columns = tfidf.get_feature_names())
'''10 is one of the features. From my webscraping, I saw reviews where people had written in
a rating, instead of having it in the rating area. So it possibly is a signal. Also star wars
is a feature.'''
dense_textonly.head()
dense_textonly['tconst'] = reviews.title

'''Realized that tconst is not a unique key. tconst plus index is. Would like to
create a review count variable if doing this over again so that the pairing of
tconst and review count is a unique key.'''

#Reading back in because need to add sentiment to reviews, and then merge the two.
dense_textonly = pd.read_csv('dense_reviews_text.csv')

dense_textonly.head(2)
dense_textonly.info()

#Adding in sentiment analysis with TextBlob.

from textblob import TextBlob

def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity

reviews['sentiment'] = reviews.text.apply(detect_sentiment)


merged_tfidf_ratings = pd.merge(reviews, dense_textonly, left_index = True, right_index = True)
merged_tfidf_ratings.info()


#Going to merge the top 100 rated IMDB movies with the TFIDF table too.

#Rename rating column in TFIDF to indiv_rating.
merged_tfidf_ratings.rename(columns = {'rating':'indiv_rating'}, inplace = True)

merged_tfidf_ratings.drop('text', axis = 1, inplace = True)
merged_tfidf_ratings.head()

top100_new.head()

top100_features = pd.merge(top100_new, merged_tfidf_ratings, how = 'right', left_on = 'tconst', right_on = 'title')
top100_features.info()
top100_features.head()

#Cleaning up summary table a bit.

top100_features.drop('tconst_y', axis = 1, inplace = True)
top100_features.drop('title_y', axis = 1, inplace = True)
top100_features.rename(columns = {'tconst_x': 'tconst', 'title_x': 'title'}, inplace = True)

#Inspecting the summary table after cleaning.

top100_features.head()



'''6. Save this merged dataframe as a csv'''

#Saving the dense tfidf as a csv...

dense_textonly.to_csv('dense_reviews_text.csv', sep = ',', encoding = 'utf-8', index_label = 'tconst')
#reviews[['title', 'rating']].to_csv('ratings_top100.csv', sep = ',', encoding = 'utf-8', index_label = 'title')

#Saving the tfidf matrix with rating per review to csv
merged_tfidf_ratings.to_csv('ratings_tfidf_reviews.csv', sep = ',', encoding = 'utf-8', index_label = 'row_num')

#Saving the merged top 100 summary and tfidf table to csv
top100_features.to_csv('top100_tfidffeatures.csv', sep = ',', encoding = 'utf-8', index_label = 'row_num')


'''Part 3: Combine Tables in PostgreSQL

1. Import your two .csv data files into your Postgre Database as two different tables
For ease, we can call these table1 and table2'''

#Did this in Python with pandas above.

'''2. Connect to database and query the joined set
​
3. Join the two tables

​
4. Select the newly joined table and save two copies of the into dataframes

​
Part 4: Parsing and Exploratory Data Analysis
1. Rename any columns you think should be renamed for clarity'''

#year_x is the year. #year_y is a feature extracted
top100_features = pd.read_csv('top100_tfidffeatures.csv')
top100_features.head()
top100_features.drop('row_num', axis = 1, inplace = True)
top100_features.columns.values



top100_features.rename(columns = {'year_x': 'release_year', 'year_y': 'year_tfidf'}, inplace = True)

top100_features.columns.values


2. Describe anything interesting or suspicious about your data (quality assurance)

'''If I am trying to look at features and their relation to individual ratings,
I have to toss out the ones with NAs. Either that, or the ones with the closest
features to the ones with missing ones could have the ratings imputed.

Except that that sounds an awful lot like prediction, and I thought I would want
individual rating to be my target.

What else is interesting or odd are the unique terms specific to certain movie
franchises - star wars, the batman terms (joker, dark knight). Also, '10' could
be an odd feature. From my inspection, some reviews had 10/10 in their text.

But it could be possible that the 10 relates to anyone who put any rating in
their review. So 10, despite being a feature of interest, according to TFIDF,
might actually not have good signal.'''

​
'''3. Make four visualizations of interest to you using the data'''

#Bar chart of counts of movies in genres. This will exceed 100 because of the
#multiple genres.
top100_new.columns.values
plt.figure(figsize = (20,10))
plt.title('Count of Genres in Top 100 Movies', fontsize = 16)
top100_new[['Action', 'Adventure', 'Animation',
       'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
       'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
       'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']].sum().plot(
       kind = 'bar', alpha = 0.3, fontsize = 14
       )
plt.savefig('genre_count_top_100.png')
#What's the text feature with the greatest importance?
plt.figure(figsize = (200,200))
top100_features[[ u'10', u'acting', u'action', u'actor',
       u'actors', u'actually', u'amazing', u'american', u'audience',
       u'away', u'bad', u'batman', u'battle', u'beautiful', u'believe',
       u'best', u'better', u'big', u'bit', u'black', u'book', u'brilliant',
       u'cast', u'character', u'characters', u'cinema', u'cinematography',
       u'classic', u'come', u'comes', u'completely', u'course', u'dark',
       u'dark knight', u'day', u'death', u'definitely', u'did', u'didnt',
       u'different', u'direction', u'director', u'does', u'doesnt',
       u'dont', u'effects', u'end', u'ending', u'epic', u'especially',
       u'excellent', u'experience', u'face', u'fact', u'family',
       u'fantastic', u'far', u'favorite', u'feel', u'felt', u'fight',
       u'film', u'films', u'finally', u'gets', u'gives', u'goes', u'going',
       u'good', u'got', u'great', u'greatest', u'guy', u'hard', u'having',
       u'help', u'hes', u'high', u'history', u'hope', u'hours', u'human',
       u'idea', u'im', u'interesting', u'isnt', u'ive', u'ive seen',
       u'jackson', u'job', u'joker', u'just', u'kind', u'king', u'knight',
       u'know', u'left', u'life', u'like', u'line', u'little', u'long',
       u'look', u'lot', u'love', u'loved', u'main', u'make', u'makes',
       u'making', u'man', u'masterpiece', u'maybe', u'men', u'mind',
       u'minutes', u'movie', u'movies', u'music', u'need', u'new',
       u'nolan', u'old', u'opinion', u'original', u'oscar', u'overall',
       u'people', u'perfect', u'performance', u'performances', u'picture',
       u'place', u'played', u'plot', u'point', u'pretty', u'probably',
       u'quite', u'read', u'real', u'really', u'reason', u'right', u'role',
       u'said', u'saw', u'say', u'scene', u'scenes', u'score', u'screen',
       u'script', u'second', u'seeing', u'seen', u'sense', u'set',
       u'shows', u'simply', u'special', u'special effects', u'star',
       u'star wars', u'start', u'story', u'sure', u'takes', u'thats',
       u'theres', u'thing', u'things', u'think', u'thought', u'time',
       u'times', u'trilogy', u'true', u'truly', u'trying', u'understand',
       u'use', u'used', u'violence', u'want', u'war', u'wars', u'wasnt',
       u'watch', u'watched', u'watching', u'way', u'wife', u'wonderful',
       u'work', u'world', u'worth', 'year_tfidf', u'years', u'young']].mean().plot(
       kind = 'bar', alpha = .3, fontsize = 15
       )
plt.title('Mean Weighted Importance of Text Terms in Reviews, using TFIDF for Importance')
plt.savefig('test_terms_importance.png')
#Relative importance of director as a text feature in reviews, and showing for
#which movies it was interesting.
top100_features.groupby('title').director.mean().sort_values(ascending = False)

#The highest importance text features in the reviews were... (top 20)

plt.figure()
top100_features[[ u'10', u'acting', u'action', u'actor',
       u'actors', u'actually', u'amazing', u'american', u'audience',
       u'away', u'bad', u'batman', u'battle', u'beautiful', u'believe',
       u'best', u'better', u'big', u'bit', u'black', u'book', u'brilliant',
       u'cast', u'character', u'characters', u'cinema', u'cinematography',
       u'classic', u'come', u'comes', u'completely', u'course', u'dark',
       u'dark knight', u'day', u'death', u'definitely', u'did', u'didnt',
       u'different', u'direction', u'director', u'does', u'doesnt',
       u'dont', u'effects', u'end', u'ending', u'epic', u'especially',
       u'excellent', u'experience', u'face', u'fact', u'family',
       u'fantastic', u'far', u'favorite', u'feel', u'felt', u'fight',
       u'film', u'films', u'finally', u'gets', u'gives', u'goes', u'going',
       u'good', u'got', u'great', u'greatest', u'guy', u'hard', u'having',
       u'help', u'hes', u'high', u'history', u'hope', u'hours', u'human',
       u'idea', u'im', u'interesting', u'isnt', u'ive', u'ive seen',
       u'jackson', u'job', u'joker', u'just', u'kind', u'king', u'knight',
       u'know', u'left', u'life', u'like', u'line', u'little', u'long',
       u'look', u'lot', u'love', u'loved', u'main', u'make', u'makes',
       u'making', u'man', u'masterpiece', u'maybe', u'men', u'mind',
       u'minutes', u'movie', u'movies', u'music', u'need', u'new',
       u'nolan', u'old', u'opinion', u'original', u'oscar', u'overall',
       u'people', u'perfect', u'performance', u'performances', u'picture',
       u'place', u'played', u'plot', u'point', u'pretty', u'probably',
       u'quite', u'read', u'real', u'really', u'reason', u'right', u'role',
       u'said', u'saw', u'say', u'scene', u'scenes', u'score', u'screen',
       u'script', u'second', u'seeing', u'seen', u'sense', u'set',
       u'shows', u'simply', u'special', u'special effects', u'star',
       u'star wars', u'start', u'story', u'sure', u'takes', u'thats',
       u'theres', u'thing', u'things', u'think', u'thought', u'time',
       u'times', u'trilogy', u'true', u'truly', u'trying', u'understand',
       u'use', u'used', u'violence', u'want', u'war', u'wars', u'wasnt',
       u'watch', u'watched', u'watching', u'way', u'wife', u'wonderful',
       u'work', u'world', u'worth', 'year_tfidf', u'years', u'young']].mean().sort_values(ascending = False)[0:20].plot(
       kind = 'bar', alpha = .3, fontsize = 15
       )
plt.title('Top 20 Text Features for Reviews')
plt.savefig('top20_text_features_tfidfranking.png')

'''Looking at the above, doesn't movie or film seem suspect? Or the plural?
It's possible it is caught out because it ends up being part of a comparison (i.e.
I have not seen a movie this bad in years.)'''

#release year distribution.
top100_new['year'].plot(kind = 'hist', alpha = 0.3)
plt.title('Distribution of Top 100 Movies by Release Year')
plt.savefig('top100_movies_release_year.png')

'''Note the skew towards recency. Could be a number of factors (i.e. greater
engagement with newer movies because of recency coinciding with increasing use
of the Internet.)'''

#What about year and rating?

top100_new.plot(kind = 'scatter', x = 'year', y = 'rating', alpha = 0.3)
plt.title('Ratings Compared to Release Year')

#Same plot as above, but with line of best fit.
plt.figure(figsize = (10,5))
sns.regplot(data = top100_new, x = 'year', y = 'rating')
plt.title('Relation of Average Rating to Release Year for Top 100 IMDB Movies, with line of best fit', fontsize = 15)
plt.savefig('rating_vs_release_year.png')


'''Interesting to see wider disperson in ratings in more recent movies compared
to before 1970.'''

#Plot of sentiment by rating...
plt.style.use('ggplot')

top100_features.boxplot(column = 'sentiment', by = 'indiv_rating')
plt.title('Sentiment by Movie Rating')
plt.xlabel('Movie Ratings')
plt.savefig('sentiment_indiv_rating.png')

sns.regplot(data = top100_features, x = 'sentiment', y = 'indiv_rating')


'''Part 5: Decision Tree Classifiers and Regressors
1. What is our target attribute?
Choose a target variable for the decision tree regressor and the classifier.'''

'''Our target data should be the individual rating.'''

'''2. Prepare the X and Y matrices and preprocess data as you see fit'''

#For each movie, I could either insert the mean rating for it or drop the missing value.

#However, I have text that I am trying to see is a determinant of the rating.
#So it might be best to drop the missing value ratings.

#Need to convert classification to dummy.
#Inspecting this, and knowing that the rating system was changed at various times,
#there should be a better proxy for classification than the rankings...
top100_features.classification = top100_features.classification.map({'Approved': 'Not_R', 'Passed': 'Not_R', 'G': 'Not_R',
'M': 'R', 'PG': 'Not_R', 'R': 'R', 'TV-14': 'Not_R', 'PG-13': 'Not_R', 'TV-MA': 'R', 'Unrated': 'Unrated',
'Not Rated': 'Unrated', 'X': 'X'})

mpaa_dummies = pd.get_dummies(top100_features.classification)

top100_features.columns.values
#Realized a column was in with no values, dropped it.
top100_features.drop('tconst.1', axis = 1, inplace = True)

top100_features.dropna(axis = 0, inplace = True)
top100_features.info()
top100_features.indiv_rating

y = top100_features.indiv_rating

features = ['release_year', 'runtime',
        'Action', 'Adventure', 'Animation',
       'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
       'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
       'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western', 'sentiment',
        u'10', u'acting', u'action', u'actor',
       u'actors', u'actually', u'amazing', u'american', u'audience',
       u'away', u'bad', u'batman', u'battle', u'beautiful', u'believe',
       u'best', u'better', u'big', u'bit', u'black', u'book', u'brilliant',
       u'cast', u'character', u'characters', u'cinema', u'cinematography',
       u'classic', u'come', u'comes', u'completely', u'course', u'dark',
       u'dark knight', u'day', u'death', u'definitely', u'did', u'didnt',
       u'different', u'direction', u'director', u'does', u'doesnt',
       u'dont', u'effects', u'end', u'ending', u'epic', u'especially',
       u'excellent', u'experience', u'face', u'fact', u'family',
       u'fantastic', u'far', u'favorite', u'feel', u'felt', u'fight',
       u'film', u'films', u'finally', u'gets', u'gives', u'goes', u'going',
       u'good', u'got', u'great', u'greatest', u'guy', u'hard', u'having',
       u'help', u'hes', u'high', u'history', u'hope', u'hours', u'human',
       u'idea', u'im', u'interesting', u'isnt', u'ive', u'ive seen',
       u'jackson', u'job', u'joker', u'just', u'kind', u'king', u'knight',
       u'know', u'left', u'life', u'like', u'line', u'little', u'long',
       u'look', u'lot', u'love', u'loved', u'main', u'make', u'makes',
       u'making', u'man', u'masterpiece', u'maybe', u'men', u'mind',
       u'minutes', u'movie', u'movies', u'music', u'need', u'new',
       u'nolan', u'old', u'opinion', u'original', u'oscar', u'overall',
       u'people', u'perfect', u'performance', u'performances', u'picture',
       u'place', u'played', u'plot', u'point', u'pretty', u'probably',
       u'quite', u'read', u'real', u'really', u'reason', u'right', u'role',
       u'said', u'saw', u'say', u'scene', u'scenes', u'score', u'screen',
       u'script', u'second', u'seeing', u'seen', u'sense', u'set',
       u'shows', u'simply', u'special', u'special effects', u'star',
       u'star wars', u'start', u'story', u'sure', u'takes', u'thats',
       u'theres', u'thing', u'things', u'think', u'thought', u'time',
       u'times', u'trilogy', u'true', u'truly', u'trying', u'understand',
       u'use', u'used', u'violence', u'want', u'war', u'wars', u'wasnt',
       u'watch', u'watched', u'watching', u'way', u'wife', u'wonderful',
       u'work', u'world', u'worth', 'year_tfidf', u'years', u'young']

X = top100_features[features]
X = X.join(mpaa_dummies)
X.dtypes


'''3. Build and cross-validate your decision tree classifier '''

#Let's train test split.

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 31) #for reproducibility.

y_test

#Helper function - cross validation with the type of scoring as an input.
def do_cross_val(model, X_data, y_data, score):
    scores = cross_val_score(model,X_data, y_data, cv = 5, n_jobs = -1, scoring = score)
    print "The %s score is %.2f +/- %.2f" % (score, scores.mean(), scores.std())
    return scores.mean(), scores.std()


#More preprocessing - scaling the numbers with RobustScaler (on the median)
#since not sure if it will help...

from sklearn.preprocessing import RobustScaler

X.dtypes

features_to_scale = ['release_year', 'runtime']
features_to_not_scale = ['Action', 'Adventure', 'Animation',
       'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
       'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
       'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western', 'sentiment',
        u'10', u'acting', u'action', u'actor',
       u'actors', u'actually', u'amazing', u'american', u'audience',
       u'away', u'bad', u'batman', u'battle', u'beautiful', u'believe',
       u'best', u'better', u'big', u'bit', u'black', u'book', u'brilliant',
       u'cast', u'character', u'characters', u'cinema', u'cinematography',
       u'classic', u'come', u'comes', u'completely', u'course', u'dark',
       u'dark knight', u'day', u'death', u'definitely', u'did', u'didnt',
       u'different', u'direction', u'director', u'does', u'doesnt',
       u'dont', u'effects', u'end', u'ending', u'epic', u'especially',
       u'excellent', u'experience', u'face', u'fact', u'family',
       u'fantastic', u'far', u'favorite', u'feel', u'felt', u'fight',
       u'film', u'films', u'finally', u'gets', u'gives', u'goes', u'going',
       u'good', u'got', u'great', u'greatest', u'guy', u'hard', u'having',
       u'help', u'hes', u'high', u'history', u'hope', u'hours', u'human',
       u'idea', u'im', u'interesting', u'isnt', u'ive', u'ive seen',
       u'jackson', u'job', u'joker', u'just', u'kind', u'king', u'knight',
       u'know', u'left', u'life', u'like', u'line', u'little', u'long',
       u'look', u'lot', u'love', u'loved', u'main', u'make', u'makes',
       u'making', u'man', u'masterpiece', u'maybe', u'men', u'mind',
       u'minutes', u'movie', u'movies', u'music', u'need', u'new',
       u'nolan', u'old', u'opinion', u'original', u'oscar', u'overall',
       u'people', u'perfect', u'performance', u'performances', u'picture',
       u'place', u'played', u'plot', u'point', u'pretty', u'probably',
       u'quite', u'read', u'real', u'really', u'reason', u'right', u'role',
       u'said', u'saw', u'say', u'scene', u'scenes', u'score', u'screen',
       u'script', u'second', u'seeing', u'seen', u'sense', u'set',
       u'shows', u'simply', u'special', u'special effects', u'star',
       u'star wars', u'start', u'story', u'sure', u'takes', u'thats',
       u'theres', u'thing', u'things', u'think', u'thought', u'time',
       u'times', u'trilogy', u'true', u'truly', u'trying', u'understand',
       u'use', u'used', u'violence', u'want', u'war', u'wars', u'wasnt',
       u'watch', u'watched', u'watching', u'way', u'wife', u'wonderful',
       u'work', u'world', u'worth', 'year_tfidf', u'years', u'young']
X_prescale = X[features_to_scale]
X_scaled = RobustScaler().fit_transform(X_prescale)
X_scaled = pd.DataFrame(X_scaled, columns = features_to_scale, index = X_prescale.index)
X_final_scaled = X_scaled.join(X[features_to_not_scale])

X_final_scaled.info()
X.info()
#Train Test Split the scaled data

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_final_scaled, y, test_size = .2, random_state = 31)

#So what is the baseline prediction?
print y.mean()
y.value_counts()

baseline_not10 = (1-y[y== 10].count()/float(y.count()))

'''There are at least two possibilities I can think of for testing with the Classifier:

 A) the y's as is, and B), relabeled classes, like y < 7 is
a dud, y = 8 is middle of the pack, and y >= 9 is a blockbuster.

2) Either toss away release_year as irrelevant (though it could impact because of fewer
people who probably saw it) or reformat it into a categorical, such as by decade.

Let's go with the variants on 1 first.'''


'''
3 - Random Forest Classifier - Unscaled Data, Y unformated.
'''
#Random Forest Classifier on y as is. Unscaled data first.

rf_class = RandomForestClassifier(random_state = 31, verbose = True, class_weight = 'balanced')
rf_class.fit(X_train, y_train)
y_pred_rfclass = rf_class.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print "The accuracy score for the unscaled features on RF Classifier is:", accuracy_score(y_test, y_pred_rfclass)
baseline_not10_test = 1 - y_test[y_test==10].count()/float(y_test.count())

print "The improvement on the baseline (essentially not 10 vs 10, since 10 is the dominant part of the data, is %.2f" % (accuracy_score(y_test, y_pred_rfclass)-baseline_not10_test)


rf_class_feature_imp = pd.DataFrame(rf_class.feature_importances_, columns = ['importance'], index = X_train.columns).sort_values(['importance'], ascending = False)
rf_class_feature_imp

'''Release year is still important, as is runtime... but these were unscaled. Have to double check with scaling...'''

#Let's also get the classification report and confusion matrix...

print np.sort(y_test.unique())

def do_class_confuse(y_t, y_p):
    print "The accuracy score for the RF Classifier is:", accuracy_score(y_t, y_p)
    print "The improvement on the baseline (essentially not 10 vs 10, since 10 is the dominant part of the data, is %.2f" % (accuracy_score(y_t, y_p)-baseline_not10_test)
    cm =  confusion_matrix(y_t, y_p)
    cf =  classification_report(y_t, y_p)

    print cf
    print cm

do_class_confuse(y_test, y_pred_rfclass)

'''The classifier is good at grouping 10 as 10, with a recall of .87. It doesn't
do well for other classes individually...'''

'''3. Random Forest Classifier - Scaled Data, Y Unformated'''

rf_class_scaled = RandomForestClassifier(random_state = 31, verbose = True, class_weight = 'balanced')
rf_class_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_rfclass_scaled = rf_class_scaled.predict(X_test_scaled)

print "The accuracy score for the scaled features on RF Classifier is:", accuracy_score(y_test_scaled, y_pred_rfclass_scaled)

print "The improvement on the baseline (essentially not 10 vs 10, since 10 is the dominant part of the data, is %.2f" % (accuracy_score(y_test_scaled, y_pred_rfclass_scaled)-baseline_not10_test)
'''Not much of an improvement.'''

pd.DataFrame(rf_class_scaled.feature_importances_, columns = ['importance'], index = X_train_scaled.columns).sort_values(['importance'], ascending = False)

'''I really think I need to try this with release_year not in the model or as a categorical.'''

do_class_confuse(y_test_scaled, y_pred_rfclass_scaled)

'''3 - Random Forest Classifier, Unscaled features, Y formated in three classes: <8, 8 and 9, 10.'''


y_test_three = y_test.map({1 : '<8', 2: '<8', 3: '<8', 4: '<8', 5: '<8',
 6: '<8', 7: '<8', 8: '8-9', 9: '8-9', 10: '10'})
y_train_three = y_train.map({1 : '<8', 2: '<8', 3: '<8', 4: '<8', 5: '<8',
 6: '<8', 7: '<8', 8: '8-9', 9: '8-9', 10: '10'})

rf_class.fit(X_train, y_train_three)
y_pred_rfclass = rf_class.predict(X_test)

do_class_confuse(y_test_three, y_pred_rfclass)

pd.DataFrame(rf_class.feature_importances_, columns = ['importance'], index = X_train.columns).sort_values(['importance'], ascending = False)

'''The feature importance seems more logical now, even though I have reservations about release year.'''

'''3 - Random Forest Classifier - Scaled Features, Y formated into three classes.'''

y_testscaled_three = y_test_scaled.map({1 : '<8', 2: '<8', 3: '<8', 4: '<8', 5: '<8',
 6: '<8', 7: '<8', 8: '8-9', 9: '8-9', 10: '10'})
y_trainscaled_three = y_train_scaled.map({1 : '<8', 2: '<8', 3: '<8', 4: '<8', 5: '<8',
 6: '<8', 7: '<8', 8: '8-9', 9: '8-9', 10: '10'})

rf_class_scaled.fit(X_train_scaled, y_trainscaled_three)
y_pred_rfclass_scaled = rf_class_scaled.predict(X_test_scaled)

do_class_confuse(y_testscaled_three, y_pred_rfclass_scaled)

'''Accuracy went down slightly...however, still interesting...'''

'''What happens if I toss out release_year?

3 - Random Forest Classifier, unscaled data, release_year out, y in three classes.'''

X_train_noyear = X_train.drop('release_year', axis = 1)
X_test_noyear = X_test.drop('release_year', axis =1)
X_test_noyear.info()

rf_class.fit(X_train_noyear, y_train_three)
y_pred_rfclass = rf_class.predict(X_test_noyear)

do_class_confuse(y_test_three, y_pred_rfclass)

'''Not an improvement to toss out release year. Let's keep it in...'''

#Let's see what cross-validating the Random Forest Classifier without optimal parameters
#returns. We'll do this with sentiment as a variable, on unscaled X features,
#with three classes for y.

do_cross_val(rf_class, X_train, y_train_three, 'accuracy')

'''4. Gridsearch optimal parameters for your classifier. Does the performance improve?
Doing this first on the unscaled data for Y split into three classes.'''

rf_class = RandomForestClassifier(random_state = 31, class_weight = 'balanced')
from sklearn.grid_search import GridSearchCV

params_rf_class = {'criterion' : ['gini', 'entropy'], 'n_estimators' : [5, 10, 20], 'max_features' : ['auto', 'log2', None], 'max_depth': [5,10,20,50], 'min_samples_leaf' : [50, 100]}

gsrf_class = GridSearchCV(rf_class, params_rf_class, n_jobs = -1, cv = 5, verbose = True)

gsrf_class.fit(X_train, y_train_three)

gsrf_class.best_params_
#That it is the max depth and min sample leaf I chose tells me that I might have wanted to specify a deeper tree.



print "The best GridSearchCV score for the classifier is %.2f, which is around the same as the non-gridsearched one." % gsrf_class.best_score_

do_class_confuse(y_test_three, gsrf_class.predict(X_test))


#Second parameter grid for gridsearch for classifier
params_rf_class = {'criterion' : ['entropy'], 'n_estimators' : [30], 'max_features' : ['auto', 'log2', None], 'max_depth': [75, 100], 'min_samples_leaf' : [10, 25]}

gsrf2_class = GridSearchCV(rf_class, params_rf_class, cv = 5, verbose = True, n_jobs = -1)

gsrf2_class.fit(X_train, y_train_three)

gsrf2_class.best_params_

gsrf2_class.best_score_

do_class_confuse(y_test_three, gsrf2_class.predict(X_test))

#Final GridSearchCV for Classifier - seeing if can improve further with more estimators.
params_rf_class = {'criterion' : ['entropy'], 'n_estimators' : [40, 50], 'max_features' : ['auto', 'log2', None], 'max_depth': [60, 75], 'min_samples_leaf' : [5, 10]}

gsrf3_class = GridSearchCV(rf_class, params_rf_class, cv = 5, verbose = True, n_jobs = -1)

gsrf3_class.fit(X_train, y_train_three)

gsrf3_class.best_params_

gsrf3_class.best_score_

do_class_confuse(y_test_three, gsrf3_class.predict(X_test))

#Fourth GridSearch, with sentiment included.

params_rf_class = {'criterion' : ['entropy'], 'n_estimators' : [50, 70, 80], 'max_features' : ['log2'], 'max_depth': [75, 100], 'min_samples_leaf' : [2, 5, 10]}

gsrf4_class = GridSearchCV(rf_class, params_rf_class, cv = 5, verbose = True, n_jobs = -1)

gsrf4_class.fit(X_train, y_train_three)

gsrf4_class.best_params_

gsrf4_class.best_score_

do_class_confuse(y_test_three, gsrf4_class.predict(X_test))

#5th Grid Search
params_rf_class = {'criterion' : ['entropy'], 'n_estimators' : [90, 100], 'max_features' : ['log2'], 'max_depth': [120, 140, 160], 'min_samples_leaf' : [2, 5]}

gsrf5_class = GridSearchCV(rf_class, params_rf_class, cv = 5, verbose = True, n_jobs = -1)

gsrf5_class.fit(X_train, y_train_three)

gsrf5_class.best_params_

gsrf5_class.best_score_

do_class_confuse(y_test_three, gsrf5_class.predict(X_test))

pd.DataFrame(gsrf5_class.best_estimator_.feature_importances_, columns = ['importance'], index = X_train.columns).sort_values('importance', ascending = False)



'''5. Build and cross-validate your decision tree regressor'''
#Unscaled Data.
rf = RandomForestRegressor(random_state = 31) #for reproducibility and seeing how long it takes.

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.cross_validation import cross_val_score

print "The r^2 for the test data with Random Forest is %.2f" % r2_score(y_test, y_pred)

do_cross_val(rf, X_train, y_train, 'r2')
'''This is a low R2. I wonder if this is because 1) unscaled numbers;
2) because release_year should be categorized by something like decade.'''

#Scaled Data
rf_scaled = RandomForestRegressor(random_state = 31, verbose = True)
rf_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = rf_scaled.predict(X_test_scaled)

print "The R^2 score for Random Forest with scaling of some data is %.2f." % (r2_score(y_test_scaled, y_pred_scaled))
'''Terrible!'''

#Cross validation of scaled data.
do_cross_val(rf_scaled, X_train_scaled, y_train_scaled, 'r2')



#Cross validation of unscaled data but with all data.
do_cross_val(rf, X, y, 'r2')
​
'''6. Gridsearch the optimal parameters for your classifier. Does performance improve?'''
#Changing the y's scale does not make sense if doing a regression.

#Part of this is honestly doing a check on which type of supervised learning makes sense.
#Using a classifier seems to make more sense.

#If I had more time, I would test this on a boosting model for regression.

params_rf_reg = {'n_estimators': [10, 20, 30, 40], 'max_features': ['auto', 'sqrt', 'log2', None], 'max_depth': [30, 50, 70], 'min_samples_leaf': [2,5,10,20]}
​
gsrf_reg = GridSearchCV(rf, params_rf_reg, n_jobs = -1, verbose = True, cv = 5)

gsrf_reg.fit(X_train, y_train)
