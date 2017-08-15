# Dimention reduction and vector representation of User transition matrix

###########################################################################################################
# Going Further into this problem of higher memory consumption in terms of Markov chain model
# to implement user transition matrix in general and user specific
# and less user specific reliable result of matrix factorization
# one can think of the neural networks as the solution to reduce the high dimentionality of the user matrix
# and built more user specific result opposed to the General Matrix factorization model
###########################################################################################################

# Here we consider the implementation similar to the word2vec where the product ids in a transaction can be considered as
# input word for which the context is the products purchased in the next transition and we can keep on training
# out user_transition_matrix till we reach the last-1 transition for products
# Dimentionality of the user transition matrix can be adjusted based on the dimenstion selected for the model
# Since the Dimentinality can be reduced memory usage for further processing will reduce
#
# considering user transition matrix of 100 dimension, memory usage taking 4byte for each point
# and for products ranging from 1-50000 ==> 50000*100*4 = approx 20mb opposed to earlier 50000*50000*4 = approx 9.3gb
# disadvantage of this is that training time for each user might take longer with low computation




