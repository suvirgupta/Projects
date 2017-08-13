import pandas as pd
import numpy as np
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Read data from files
train = pd.read_csv( r"C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\labeledTrainData.tsv", header=0, delimiter="\t",quoting=3 )
test = pd.read_csv( r"C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( r"C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
def preprocessing(reviews):
## remove the Html Tags
    review_text = BeautifulSoup(reviews).get_text()
## Remove non letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
## set all words to lower digits
    review_text = review_text.lower().split()
## remove stopwords
    stopword = set(stopwords.words("english"))
    words = [w for w in review_text if w not in stopword]
### lemmatization of the words to reduce them to their standard foramat
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    return words

import nltk.data


# # Define a function to split a review into parsed sentences
# def review_to_sentences( review, tokenizer):
# # Function to split a review into parsed sentences. Returns a
# # list of sentences, where each sentence is a list of words
# #
# # 1. Use the NLTK tokenizer to split the paragraph into sentences
#     raw_sentences = tokenizer.tokenize(review.strip())
# #
# # 2. Loop over each sentence
#     sentences = []
#     for raw_sentence in raw_sentences:
#     # If a sentence is empty, skip it
#         if len(raw_sentence) > 0:
#     # Otherwise, call review_to_wordlist to get a list of words
#             sentences.append( preprocessing( raw_sentence))
# #
# # Return the list of sentences (each sentence is a list of words,
# # so this returns a list of lists
#     return sentences

#
# # Load the punkt tokenizer
# sentences = []
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# for review in train["review"]:
#     sentences += review_to_sentences(review, tokenizer)

review_list=[" ".join(preprocessing(review)) for review in train["review"]]
review_list_val = [" ".join(preprocessing(review)) for review in test["review"]]


def tokenizer(review_list, sentiment):
## Taking the 50000 most frequently occuring words
## built object of the class Tokenizer
    tokenizer = Tokenizer(50000)
### fit int index to each world of the text
    tokenizer.fit_on_texts(review_list)
### convert list of string to list of list and list containing sequence of indexes of the words in the string
    sequences = tokenizer.texts_to_sequences(review_list)
## index generated for each word
    word_index = tokenizer.word_index

## padding the index words in one document to be of similar size
## padding with zeroes
    dat = pad_sequences(sequences)
    labels = np.asarray(sentiment)

    indices = np.arange(dat.shape[0])
    np.random.shuffle(indices)
    data = dat[indices]
    labels = labels[indices]
    nb_validation_samples = int(.1 * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train,y_train,x_val,y_val,word_index

x_train,y_train,x_val,y_val, word_index= tokenizer(review_list, train["sentiment"])

max_seq_len = len(x_train[1])

#### importing glove vectors from the nltk library
#### vector representation of the words are given onthe glove webpage hosted by Stanford
#### we are taking a text file of 100 dimentional vector and converting to dictionary of words and vectors

## download file using urllib.request.urlretrieve
import urllib
filename, _ = urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip', 'glove.6B.zip')

import os
statinfo = os.stat(filename)
statinfo.st_size


from zipfile import ZipFile as zp
f=zp(filename)
data = []
for file in f.namelist():
    if file == 'glove.6B.100d.txt':
        with f.open(file) as df:
            [data.append(x) for x in df]



# data1 = [x.split(b' ') for x in data]
# str(data1[1][1])
data = [x.decode('utf-8 ') for x in data]
data = [x.split(' ') for x in data]

data[0]
### convert the list of strings to the dictionary format

glove_dict = {}
for item in data:
    word = item[0]
    vector = np.asarray(item[1:],dtype = 'float32')
    glove_dict[word]= vector


embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = glove_dict.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



from keras.layers import Embedding, Input
from keras.models import Sequential
from keras.models import  Model
from keras.layers import Conv1D, Dense,Flatten,MaxPooling1D
embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length= max_seq_len,
                            trainable=False)
# model = Sequential()
#
# sequence_input = Input(shape=(max_seq_len,), dtype='int32')
# model.add(embedding_layer(sequence_input))
# ### adding convolution1D layer
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(35))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='softmax'))
# model.compile(optimizer = 'rmsprop',loss='binary_crossentropy', metrics = 'accuracy' )

### sequencial input of length of array
sequence_input = Input(shape=(max_seq_len,), dtype='int32')
#### Emedded layer with the input
embedded_sequences = embedding_layer(sequence_input)
## output of embedded layer to convolution 1d
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
# output dense layer with output as 1 i.e 0,1

preds = Dense(1 , activation='sigmoid')(x)

model = Model(sequence_input, preds)
### use loss as binary_crossentropy   for binary categorical variables
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=5, batch_size=128)