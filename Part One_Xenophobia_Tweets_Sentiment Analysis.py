# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:50:39 2023

@author: pc
"""
###########################################################################################
######################## THIS IS THE FIRST PART OF THE PROJECT ############################
###########################################################################################


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import csv
from nltk.stem.porter import *
from textblob import TextBlob
import nltk
import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

#%%

######################################### STEP ONE ########################################

#############################
### Explore the whole dataset
#############################

''' 
I am using the total dataset (over 4.7 millions records)
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

###################################### STEP TWO #######################################
################ Now only use the negative subset to do more analysis #################

######################
### Text Preprocessing 
######################

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

'''
Inspired By CHRISTIAN LILLELUND's Kaggle post, he thinks that xenophobia tweets naturally 
belong to negative tweets. 
'''

##############################
### Find all Xenophobia speech
##############################


# Load the preprocessed data
neg_tweets = pd.read_csv('preprocessed_data.csv', encoding='latin-1')

# Split the text into sentences
sentences = []
for text in neg_tweets['text']:
    if isinstance(text, str):
        sentences += nltk.sent_tokenize(text)
    else:
        sentences += nltk.sent_tokenize(str(text))
        

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
# neg_tweets['Xenophobic'] = neg_tweets.text.apply(lambda x: is_tweet_xenophobic(x.text) if type(x.text)==str else False)
# Apply the function to create a new column with the boolean value indicating xenophobia
neg_tweets['text'] = neg_tweets['text'].fillna('')
neg_tweets['Xenophobic'] = neg_tweets['text'].apply(lambda x: is_tweet_xenophobic(x))
neg_tweets[['Xenophobic']] = neg_tweets[['Xenophobic']].fillna(value=False)

counts = neg_tweets['Xenophobic'].value_counts()
print(counts)

'''
Among 928612 negative tweets, there are 52428 tweets are xenophobic.
'''

########################################
### Let's visualize the whole data again
########################################

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


##############################
### Prepare for Topic Modeling
##############################

'''
Since I only have 52428 records of Xenophobic data, I plan to map them from the original data set. 
And then do further analysis only on these tweets
https://medium.com/towards-data-science/lda-topic-modeling-with-tweets-deff37c0e131
'''


xenophobic = neg_tweets.loc[neg_tweets['Xenophobic'] == True]
xenophobic = xenophobic.drop(['Unnamed: 0', 'blob_sentiment', 'Xenophobic'], axis=1)
len(xenophobic)
xenophobic.info()

xenophobic.to_csv('xenophobic tweets.csv')

