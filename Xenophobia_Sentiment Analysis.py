# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:50:39 2023

@author: pc
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import re
import csv
from nltk.stem.porter import *
from textblob import TextBlob
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from nltk.probability import FreqDist
import gensim
from gensim.utils import simple_preprocess
# Load English language model
nlp = spacy.load("en_core_web_sm")

#%%

####################### STEP ONE ##########################

###
### Explore a subset of the dataset
### Only 500000 records used here
### Get familiar with my data set
###

nrows = 500000
df = pd.read_csv('Xenophobia.csv', nrows=nrows, encoding='latin1')

# Drop columns not used for modelling
cols_to_drop = ['status_id', 'created_at', 'location', 'favorite_count', 'retweet_count', 'followers_count', 'friends_count', 'statuses_count']
df.drop(cols_to_drop, axis=1, inplace=True)
            
# Convert text to string type
df['text'] = df['text'].astype(str)

print("Total number of samples:", len(df))

df.head()

###
### Sentiment Analysis by using TextBlob 
###

def blob_prediction(text):
    pred = TextBlob(text).sentiment.polarity
    if pred<0:
        return -1
    elif pred>0:
        return 1
    else:
        return 0
    
df['blob_sentiment'] = df['text'].apply(lambda x: blob_prediction(x))

print(f"Number of positive tweets: {len(df.loc[df['blob_sentiment'] == 1])}")
print(f"Number of negative tweets: {len(df.loc[df['blob_sentiment'] == -1])}")
print(f"Number of neutual tweets: {len(df.loc[df['blob_sentiment'] == 0])}")

'''
Number of positive tweets: 226135
Number of negative tweets: 100042
Number of neutual tweets: 173823
'''

###
### Sentiment Analysis by using Vader
###

sid = SentimentIntensityAnalyzer()
def vader_prediction(text):
    pred = sid.polarity_scores(text)
    mx = max([pred['neg'], pred['neu'], pred['pos']])
    if pred['neu'] == mx:
        return 0
    elif pred['neg'] == mx:
        return -1
    elif pred['pos'] == mx:
        return 1
    else:
        return 0
    
df['vader_sentiment'] = df['text'].apply(lambda x: vader_prediction(x))

print(f"Number of positive tweets: {len(df.loc[df['vader_sentiment'] == 1])}")
print(f"Number of negative tweets: {len(df.loc[df['vader_sentiment'] == -1])}")
print(f"Number of neutual tweets: {len(df.loc[df['vader_sentiment'] == 0])}")

'''
Number of positive tweets: 19198
Number of negative tweets: 10327
Number of neutual tweets: 470475
'''

###
### Visualize my results
###

# Sentiment analysis data using TextBlob
pos1 = len(df.loc[df['blob_sentiment'] == 1])
neg1 = len(df.loc[df['blob_sentiment'] == -1])
neu1 = len(df.loc[df['blob_sentiment'] == 0])

# Sentiment analysis data using Vader
pos2 = len(df.loc[df['vader_sentiment'] == 1])
neg2 = len(df.loc[df['vader_sentiment'] == -1])
neu2 = len(df.loc[df['vader_sentiment'] == 0])

# Data for pie charts
labels = ['Positive', 'Negative', 'Neutral']
sizes1 = [pos1, neg1, neu1]
colors1 = ['yellowgreen', 'lightcoral', 'gold']
sizes2 = [pos2, neg2, neu2]
colors2 = ['lightblue', 'lightsalmon', 'lavender']

# Create subplots for two pie charts
plt.rcParams['figure.dpi'] = 300
fig, (ax1, ax2) = plt.subplots(1, 2)

# Create first pie chart
ax1.pie(sizes1, labels=labels, colors=colors1, autopct='%1.1f%%', startangle=90)
ax1.set_title('TextBlob')

# Create second pie chart
ax2.pie(sizes2, labels=labels, colors=colors2, autopct='%1.1f%%', startangle=30, labeldistance=1.1, pctdistance=0.8)
ax2.set_title('Vader')

# Add title for the figure
fig.suptitle('Comparison of Sentiment Analysis Results')
plt.tight_layout()
fig.savefig('sentiment analysis.png')
# Show the figure
plt.show()


#%%

####################### STEP TWO ##########################

###
### Explore the whole dataset
###

''' 
Now I am using the total dataset (over 4.7 millions records)
to do analysis. Especially focus on those negative tweets classified by sentiment analysis. 
'''

ah = pd.read_csv('Xenophobia.csv', encoding='latin1')

# Drop columns not used for modelling
cols_to_drop = ['status_id', 'created_at', 'location', 'favorite_count', 'retweet_count', 'followers_count', 'friends_count', 'statuses_count']
ah.drop(cols_to_drop, axis=1, inplace=True)
            
# Convert text to string type
ah['text'] = ah['text'].astype(str)

print("Total number of samples:", len(ah))

ah['blob_sentiment'] = ah['text'].apply(lambda x: blob_prediction(x))

print(f"Number of positive tweets: {len(ah.loc[ah['blob_sentiment'] == 1])}")
print(f"Number of negative tweets: {len(ah.loc[ah['blob_sentiment'] == -1])}")
print(f"Number of neutual tweets: {len(ah.loc[ah['blob_sentiment'] == 0])}")

'''
Number of positive tweets: 2176284
Number of negative tweets: 928612
Number of neutual tweets: 1663513
'''

# Your sentiment analysis data
positive_tweets = 2176284
negative_tweets = 928612
neutral_tweets = 1663513

# Data for pie chart
plt.rcParams['figure.dpi'] = 300
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_tweets, negative_tweets, neutral_tweets]
colors = ['yellowgreen', 'lightcoral', 'gold']

# Create pie chart
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# Add title
plt.title('Sentiment Analysis Results')
plt.tight_layout()
plt.savefig("Sentiment Anlysis by using TextBlob.png")
# Show the chart
plt.show()

    
#%%

####################### STEP THREE ##########################
### Now only use the negative subset to do more analysis ####

###
### Text Preprocessing 
###

neg_ah = ah.loc[ah['blob_sentiment'] == -1]
len(neg_ah)
neg_ah.to_csv('Neg_Tweets.csv')

filename = "Neg_Tweets.csv"

# Define the pattern to remove Twitter handles (@username)
handle_pattern = r'@[\w]+'

# Open the file in read mode
with open(filename, 'r', encoding='latin1') as fh:
    # Create a CSV reader that reads each row as a dictionary
    reader = csv.DictReader(fh)
    
    # Open a new file to write the preprocessed data to
    with open('preprocessed_data.csv', 'w', newline='', encoding='latin1') as out_fh:
        # Create a CSV writer that writes each row as a dictionary
        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        stemmer = PorterStemmer()
        # Iterate over each row in the file
        for row in reader:
            # Preprocess the text data
            text = str(row['text'])
            # Remove Twitter handles
            text = re.sub(handle_pattern, '', text)
            # Remove punctuations, numbers, and special characters
            text = re.sub(r'[^\w\s]', '', text)  
            #Lowercase text
            text = text.lower()
            # Remove all words below 3 characters
            text = ' '.join([w for w in text.split() if len(w)>3])
            # Tokenize the tweets
            tokenized_tweet = text.split()
            ## Stem the tweets
            #tokenized_tweet = [stemmer.stem(i) for i in tokenized_tweet]
            
            tokenized_tweet = ' '.join(tokenized_tweet)
            # Update the row with the preprocessed text data
            row['text'] = tokenized_tweet
            
            # Write the row to the new file
            writer.writerow(row)

#%%

###
### WordCloud the processed negative words. 
###

neg_tweets = pd.read_csv('preprocessed_data.csv',  encoding='latin-1')

# Join the text together
long_string = ','.join([str(item) for item in neg_tweets['text'].fillna("")])

# Create a WordCloud object
wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3, contour_color='steelblue', max_font_size=80)

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()


###
### Another try of wordcloud
###

# Load the preprocessed data
neg_tweets = pd.read_csv('preprocessed_data.csv', encoding='latin-1')

# Split the text into sentences
sentences = []
for text in neg_tweets['text']:
    if isinstance(text, str):
        sentences += nltk.sent_tokenize(text)
    else:
        sentences += nltk.sent_tokenize(str(text))
        
font_path = "arial.ttf"
# Generate word clouds for each sentence
for i in range(0, len(sentences), 1000):
    long_string = ','.join([str(item) for item in sentences[i:i+1000]])
    wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3, contour_color='steelblue', max_font_size=80, font_path = font_path)
    wordcloud.generate(long_string)
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'wordcloud_{i}.png')


#%%

'''
Inspired By CHRISTIAN LILLELUND's Kaggle post, he thinks that xenophobia tweets naturally 
belong to negative tweets. 
'''

###
### Find all Xenophobia speech
###

# Helper function to match xenophoic tweets
def is_tweet_xenophobic(tweet):
    for s in search_terms:
        if s in tweet:
            return True
    return False

# Define search terms that appear in xenophic tweets
search_terms = ['alien', 'asian', 'kung flu', 'anti-asian', 
                'china', 'chinese', 'wuhan virus', 'criminal', 
                'floater', 'foreigner', 'greenhorn', 'aliens', 'foreigners',
                'illegal', 'intruder', 'invader', 'migrant', 'invaders',
                'immigrants', 'newcomer', 'odd one out', 'outsider', 'outsiders',
                'refugee', 'newcomers', 'send her back', 'refugees',
                'send him back', 'send them back', 'settler', 'stranger',
                'illegal aliens', 'china virus', 'refugee', 'settlers',
                'strangers', 'migrants', 'criminals']

# Find all xenophobic tweets looking only at the negative tweets
#neg_tweets['Xenophobic'] = neg_tweets.text.apply(lambda x: is_tweet_xenophobic(x.text) if type(x.text)==str else False)
# Apply the function to create a new column with the boolean value indicating xenophobia
neg_tweets['text'] = neg_tweets['text'].fillna('')
neg_tweets['Xenophobic'] = neg_tweets['text'].apply(lambda x: is_tweet_xenophobic(x))
neg_tweets[['Xenophobic']] = neg_tweets[['Xenophobic']].fillna(value=False)

counts = neg_tweets['Xenophobic'].value_counts()
print(counts)

'''
Among 928612 negative tweets, there are 52428 tweets are xenophobic.
'''

###
### Let's visualize the whole data again
###

# Your sentiment analysis data
positive_tweets = 2176284
neutral_tweets = 1663513
negative_tweets = 876184
xenophobic_tweets = 52428

# Data for pie chart
plt.rcParams['figure.dpi'] = 300
labels = ['Positive', 'Neutral', 'Negative', 'Xenophobic']
sizes = [positive_tweets, neutral_tweets, negative_tweets, xenophobic_tweets]
colors = ['yellowgreen', 'gold', 'lightcoral', 'maroon']

# Create pie chart
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# Add title
plt.title('Sentiment Analysis Results')
plt.tight_layout()
plt.savefig("Xenophobic Analysis.png")
# Show the chart
plt.show()


#%%


####################### STEP FOUR ##########################
#################### Topic Modeling ########################

'''
Since I only have 52428 records of Xenophobic data, I plan to map them from the original data set. 
The do more deep analysis on these tweets
https://medium.com/towards-data-science/lda-topic-modeling-with-tweets-deff37c0e131
'''

xenophobic = neg_tweets.loc[neg_tweets['Xenophobic'] == True]
xenophobic = xenophobic.drop(['Unnamed: 0', 'blob_sentiment', 'Xenophobic'], axis=1)
len(xenophobic)
xenophobic.info()

all_words = [word for tokens in xenophobic['text'] for word in tokens]
tweet_lengths = [len(tokens) for tokens in xenophobic['text']]
vocab = sorted(list(set(all_words)))

print('{} words total, with a vocabulary size of {}'.format(len(all_words), len(vocab)))
print('Max tweet length is {}'.format(max(tweet_lengths)))
#7350328 words total, with a vocabulary size of 49
#Max tweet length is 660

###
### Preprocessing Data
###
# remove punctuation --> removeing as I believe punctuation is beneficial in LDAs
xenophobic['processed_data']=xenophobic['text'].map(lambda x: re.sub('[;,\.!?&]','',x))

xeno_data = xenophobic.text.values.tolist()
data_words = list(xeno_data)
print(data_words[:1][0])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=1, threshold=5) 
# higher threshold fewer phrases.
# Faster way to get a sentence blubbed as a trigram/bigram.
bigram_mod=gensim.models.phrases.Phraser(bigram)
trigram_mod=gensim.models.phrases.Phraser(trigram)

#%%
###
### There are three weired characters: ãââ, let's remove them.
###


plt.rcParams['figure.dpi'] = 300
plt.figure(figsize = (15,8))
sns.countplot(tweet_lengths)
plt.title('Tweet Length Distribution', fontsize = 18)
plt.xlabel('Words per Tweet', fontsize = 12)
plt.ylabel('Number of Tweets', fontsize = 12)
plt.tight_layout()
plt.savefig('xenophobic counts.png')
plt.show()


#iterate through each tweet, then each token in each tweet, and store in one list
flat_words = [item for sublist in xenophobic['text'] for item in sublist]
word_freq = FreqDist(flat_words)
word_freq.most_common(30)

