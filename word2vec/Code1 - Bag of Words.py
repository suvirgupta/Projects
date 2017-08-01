# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author: Suvir

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


train = pd.read_csv('C:/Users/suvir/Downloads/labeledTrainData1.csv', header=0)

t1 = train.ix[0:20000,]
t2 = pd.read_csv('C:/Users/suvir/Downloads/labeledTrainData2.csv', header=0)
t1.shape
t2.shape


# Printing of size structure of Traning data 
print('Dimension of Labeled Training Data: {}.'.format(t1.shape))
print('There are {0} samples and {1} variables in the training data.'.format(t1.shape[0], t1.shape[1]))
display(t1.head())

print(t1.review[0])

t1['review_bs'] = t1['review'].apply(lambda x: BeautifulSoup(x, 'html.parser'))
t1.review_bs[0].get_text()
t1['review_letters_only'] = t1['review_bs'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x.get_text()))
t1['review_letters_only'][0]
t1['review_words'] = t1['review_letters_only'].apply(lambda x: x.lower().split())


t1['review_words'][0]

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. This will Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. This will Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. This will Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    #
    stops = set(stopwords.words("english"))                  
    # 
    # 5. This will Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

cleaned_review = review_to_words( t1["review"][0] )
print cleaned_review

# Get the number of reviews based on the dataframe column size
num_reviews = t1["review"].size

# Initialize an empty list to hold the clean reviews
cleaned_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    cleaned_train_reviews.append( review_to_words( t1["review"][i] ) )
    
print "Cleaning and parsing the training set movie reviews...\n"
cleaned_train_reviews = []
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )                                                                    
    cleaned_train_reviews.append( review_to_words( t1["review"][i] ))
    


print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(cleaned_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print vocab


# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag
    

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, t1["sentiment"] )

t1.shape
train.shape
t2.shape
# Create an empty list and append the clean reviews one by one

num_reviews = len(t2["review"])
cleaned_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    cleaned_review = review_to_words( t2["review"][i] )
    cleaned_test_reviews.append( cleaned_review )

# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(cleaned_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
train_predict = forest.predict(test_data_features)



# Read the test data

test = pd.read_csv("C:/Users/suvir/Downloads/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )


outcome = pd.read_csv("C:/Users/suvir/Downloads/Bag_of_Words_model_prediction.csv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
cleaned_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    cleaned_review = review_to_words( test["review"][i] )
    cleaned_test_reviews.append( cleaned_review )

# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(cleaned_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)


# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )


# Use pandas to write the comma-separated output file
output.to_csv( "C:/Users/ragha/Downloads/Bag_of_Words_model.csv", index=False, quoting=3 )


       
trainresult = forest.predict(train_data_features)
train_predict

output = pd.DataFrame( data={"id":train["id"], "sentiment":train_predict} )



# Use pandas to write the comma-separated output file
output.to_csv( "C:/Users/ragha/Downloads/Bag_of_Words_model1.csv", index=False, quoting=3 )


# Visualization and Fit Statistics 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics

actualout = np.asarray((t2.ix[0:50001,1]))

# ROC Value 
roc_auc = roc_auc_score(actualout,pd.to_numeric(train_predict))
roc_auc


# Confusion Matrix Generation 
df_confusion = pd.crosstab(actualout, train_predict)
df_confusion

from pandas_ml import ConfusionMatrix

# Confusion Matrix Generation 
confusion_matrix = ConfusionMatrix(actualout, pd.to_numeric(train_predict))

print("Confusion matrix:\n%s" % confusion_matrix)

import matplotlib.pyplot as plt

confusion_matrix.plot()

cm  = ConfusionMatrix(actualout, pd.to_numeric(train_predict))

cm.print_stats()




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
