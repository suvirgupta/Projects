# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:34:26 2017

@author: raghav
"""

import os
import pandas as pd
import numpy as np
from IPython.display import display

from bs4 import BeautifulSoup
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Read data from files 
train = pd.read_csv('C:/Users/ragha/Downloads/labeledTrainData1.csv', header=0)

t1 = train.ix[0:20000,]
t1.shape

t2 = pd.read_csv('C:/Users/ragha/Downloads/labeledTrainData2.csv', header=0)
t2.shape

test = pd.read_csv( "C:/Users/ragha/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3 )
test.shape

unlabeled_train = pd.read_csv( "C:/Users/ragha/Downloads/unlabeledTrainData.tsv", 
                              header=0, delimiter="\t", quoting=3 )
unlabeled_train.shape
# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (t1["review"].size,  
 test["review"].size, unlabeled_train["review"].size )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (t1["review"].size,  
 test["review"].size, unlabeled_train["review"].size )

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review,'lxml').get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in t1["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

# Check how many sentences we have in total - should be around 850,000+
print len(sentences)
print sentences[0]
print sentences[1]

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
import datetime
print "Training model..."
startTime = datetime.datetime.now()
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
print datetime.datetime.now() - startTime
                  
# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)



from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")

type(model.wv.syn0)

model.wv.syn0.shape

model["flower"]

import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
for review in t1["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )


print "Creating average feature vecs for training Test Set reviews"
clean_train_predict_reviews = []
for review in t2["review"]:
    clean_train_predict_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

train_predictDataVecs = getAvgFeatureVecs(clean_train_predict_reviews, model, num_features )


print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )


# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, t1["sentiment"] )

# Training-Test set  & extract results
train_predict = forest.predict(train_predictDataVecs )

# Test & extract results 
result = forest.predict( testDataVecs )


# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

output.to_csv( "C:/Users/ragha/Downloads/Word2Vec_model_prediction.csv", index=False, quoting=3 )


# Visualization and Fit Statistics 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics

actualout = np.asarray((t2.ix[0:50001,1]))

# ROC Value 
roc_auc = roc_auc_score(actualout,pd.to_numeric(train_predict))
roc_auc

from pandas_ml import ConfusionMatrix

# Confusion Matrix Generation 
confusion_matrix = ConfusionMatrix(actualout, pd.to_numeric(train_predict))

print("Confusion matrix:\n%s" % confusion_matrix)

import matplotlib.pyplot as plt

confusion_matrix.plot()

cm  = ConfusionMatrix(actualout, pd.to_numeric(train_predict))

cm.print_stats()

binary_confusion_matrix.plot()
plt.show()
df_confusion = pd.crosstab(actualout, train_predict)
df_confusion

# AUC Curve -------------------------

fpr, tpr, threshold = metrics.roc_curve(actualout, pd.to_numeric(train_predict))
roc_auc = metrics.auc(fpr, tpr)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




#Clustering 

from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.wv.index2word, idx ))


# For the first 10 clusters
for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print words




df = pd.DataFrame(word_centroid_map) 



### Multidimensional scalling to reduce multidimension to two dimension for cluster Visualization
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(model.wv.syn0) # finding cosine differences for the vectors

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]















 6567def create_bag_of_centroids( wordlist, word_centroid_map ):/*986532
                                
                                
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,t1["sentiment"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "C:/Users/ragha/Downloads/BagOfCentroids.csv", index=False, quoting=3 )