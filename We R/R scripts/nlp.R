# cleaning the global environment to get started with the project
rm(list = ls())
# installing all the required packages mentioned below
# wordcloud: used for constructing the wordcloud
install.packages("wordcloud")
# text mining: used for text mining purposes
install.packages("tm")
# text2vec -> used for vectorization purposes
install.packages("text2vec")
# SnowballC: used for performing the stemming operations
install.packages("SnowballC")
# data.table: to enable the fast table read
install.packages("data.table")

# loading the data.table library
library(data.table)

# loading the input data present in Combined_News_DJIA.csv file
Combined_News_DJIA <- fread("C:/workspace_r/data/Combined_News_DJIA.csv")

# viewing the recently loaded data, for confirmation
View(Combined_News_DJIA)

# creating documents out of all the top 25 news for each day
# combining all the 25 columns in to a single column
Combined_News_DJIA$document <- paste(Combined_News_DJIA$Top1,Combined_News_DJIA$Top2,Combined_News_DJIA$Top3,Combined_News_DJIA$Top4,
                                     Combined_News_DJIA$Top5,Combined_News_DJIA$Top6,Combined_News_DJIA$Top7,Combined_News_DJIA$Top8,
                                     Combined_News_DJIA$Top9,Combined_News_DJIA$Top10,Combined_News_DJIA$Top11,Combined_News_DJIA$Top12,
                                     Combined_News_DJIA$Top13,Combined_News_DJIA$Top14,Combined_News_DJIA$Top15,Combined_News_DJIA$Top16,
                                     Combined_News_DJIA$Top17,Combined_News_DJIA$Top18,Combined_News_DJIA$Top19,Combined_News_DJIA$Top20,
                                     Combined_News_DJIA$Top21,Combined_News_DJIA$Top22,Combined_News_DJIA$Top23,Combined_News_DJIA$Top24,
                                     Combined_News_DJIA$Top25,sep=' ')

# performing the date format convertion for analysis and further operations
Combined_News_DJIA$Date_f<- as.Date(strptime(Combined_News_DJIA$Date,'%m/%d/%Y'))
# extracting the actual data based on the column indices
data<- Combined_News_DJIA[,c(29,2,28)]
# extracting the date information
data$Date_f<- format(data[,1],'%Y-%m')
# extracting the month information
month <- function(x)format(x, '%Y-%m')
# performing the aggregation
data= aggregate(document~Date_f,data[,c(1,3)],paste,collapse = " ")


# loading the SnowballC library
library(SnowballC)

# loading the text2vec library
library(text2vec)

# creating a function for the stemming operation on the news
# input is the news
stem_tokenizer = function(x){
  # creating the docunized documents from the news element
  token = word_tokenizer(x)
  
  # the stemming operation is performed with the help of 
  # Porter's word stemming algorithm which is passed in the
  # language attribute
  return(lapply(token,SnowballC::wordStem,language="porter"))
}

# performing the stemming operation on this data
data$document =stem_tokenizer(data$document)

# viewing the newly created preprocessed data
View(data)

# function for doing the DATA PREPROCESSING
# input is the top 25 news
preprocess = function(data)
{
  # removing the pipe character from the news
  data$document <- gsub('b"|b\'|\\\\|\\"', "", data$document)
  
  # removing the special characters from the news
  data$document <- gsub("([<>])|[[:punct:]]", "\\1", data$document)
  
  # removing the digits present in the news
  data$document <- gsub("[[:digit:]]", "", data$document)
  
  # remove the html links present in the news
  data$document = gsub("http\\w+", "", data$document)
  
  # trimming the news so that unnecessary spaces will be removed
  data$document = gsub("[ \t]{2,}", "", data$document)
  data$document = gsub("^\\s+|\\s+$", "", data$document)
  
  # converting all the characters to lower case letters
  # for the purpose of NLP and text analysis
  data$document = tolower(data$document)
  
  # returning the refined data object, after removing 
  # all unnecessary parts from it
  return(data)
}

# caling the preprocessing function to preprocess the data
data = preprocess(data)

# loading the text mining library
library(tm)

# creating function for removing the stop words from the news
# input is the news strring and the stop words
rm_words <- function(string, words) {
  # looking for a match of the stop words in the news 
  stopifnot(is.character(string), is.character(words))
  
  # creating a splitted array of all the words in each news
  # fixed = TRUE is given for a faster operation
  spltted <- strsplit(string, " ", fixed = TRUE)
  
  # applying the stop word removal operation to each element in the news
  vapply(spltted, function(x) paste(x[!tolower(x) %in% words], collapse = " "), character(1))
}

# calling the stop word removal function to the actual news
data$document = rm_words(data$document, tm::stopwords("en"))

# use data.table fuction for fast indexing hence faster results compared to traditional subseting
data<- setkey(data,Date_f)

# viewing the newly created preprocessed data
View(data)

# creating the training dataset
data_train <- data[data$Date_f<="2014-12-31",]

# creating the test dataset
data_test  <- data[data$Date_f>"2014-12-31",]

# loading this text2vec package
library(text2vec)

# creating a function for performing Vectorization operation
# input is an individual token from the news
vectorise = function(token){
  # performing the tokenization operation
  it_train = itoken(token$document_token, 
                    ids = token$Date_f, 
                    progressbar = FALSE)
  
  # creating vocabulary words 
  vocab = create_vocabulary(it_train,ngram = c(1L,5L))
  
  # creating the vectorizer object to create the DTM
  vectorizer = vocab_vectorizer(vocab)
  
  # creating the DTM
  dtm_train = create_dtm(it_train, vectorizer)
  
  # defining  tfidf model
  tfidf = TfIdf$new()
  
  # fit model to train data and transform train data with fitted model
  dtm_train_tfidf = fit_transform(dtm_train, tfidf)
  
  # returning the newly created DTM TFIDF object
  return(dtm_train_tfidf)
}

# creating the vectorization object for the training dataset
train_vec = vectorise(data_train)

# creating the vectorization object for the test dataset
test_vec = vectorise(data_test)

lda_it_train = itoken(data$document, 
                    ids = data$Date_f, 
                    progressbar = FALSE)
lda_vocab = create_vocabulary(lda_it_train,ngram = c(1L,2L))
vectorizer = vocab_vectorizer(lda_vocab)
lda_dtm_train = create_dtm(lda_it_train, vectorizer)
tfidf = TfIdf$new()

# fit model to train data and transform train data with fitted model
lda_dtm_train_tfidf = fit_transform(lda_dtm_train, tfidf)

# LDA MODEL
# Train LDA model to fit the words in corpus to set no of topics defined
lda_model = 
  LatentDirichletAllocation$new(n_topics = 15, vocabulary = lda_vocab, 
          doc_topic_prior = 0.1, topic_word_prior = 0.01)
## train document term matrix to allocate optimum no of topic percentage to documents
doc_topic_distr = 
  lda_model$fit_transform(lda_dtm_train, n_iter = 1000, convergence_tol = 0.01, 
                          check_convergence_every_n = 10)

# installing all the required packages mentioned below
install.packages("LDAvis")
# loading this text2vec package
library(LDAvis)

# plotting interactive fitted LDA model
x = lda_model$plot()
lda_model$get_word_vectors()

# getting the range of the fitted model
range(doc_topic_distr[,3])

# getting the date information for the required cluster
doc_topic_distr[doc_topic_distr[,3]>950,3]

# Performing LATENT SEMANTIC ANALYSIS
lsa = LatentSemanticAnalysis$new(15)
d1 = lsa$fit_transform(lda_dtm_train)
d2 =  lsa$fit_transform(t(as.matrix(lda_dtm_train)))
## finding index with the known words and getting weights for each topic
d1["2008-09",] # Document topic matrix
d2["financial_crisis",]  # Words topic matrix

####kmean not good result
library(stats)
clustmean <-kmeans(t(as.matrix(lda_dtm_train)),15)

# append cluster assignment
mydata <- data.frame(t(as.matrix(lda_dtm_train)), clustmean$cluster)
datanames=row.names(mydata)
wordgroup= cbind(datanames,mydata$clustmean.cluster)

## best option to analyse document term matrix
DTM_frame = data.frame(t(as.matrix(lda_dtm_train)))
(DTM_frame["financial_crisis",])
(DTM_frame["referendum",])
(DTM_frame["brexit",])
(DTM_frame["china",])
(DTM_frame["bankruptcy",])
(DTM_frame["great_depression",])
(DTM_frame["financial_crisis",])         
