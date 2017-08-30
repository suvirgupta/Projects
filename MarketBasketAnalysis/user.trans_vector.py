# Using word vectorization technique to find out product in context of a product purchased in one transaction
# Dimention reduction and vector representation of User transition matrix
# based on word vectorization and Miklov model of vectorization and finding cosine similarity in vectors to find the context of product
#in sequence
## paper:http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
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
import numpy as np
import tensorflow as tf

prod_seq= '1511 18523 9934 47209 44473 4920 13740 40910 21290 21137 44275 49235 28204 4793 33497 10895 41950 25072 48230 28993 5194 38313 42383 15700 5077 39921 2450 35108 31263 43789 21903 41544 18610 49174 21461 30336 39928 37220 17979|1511 21137 5194 2770 44275 4920 13740 47209 6631 38313 25072 4793 28204 21290 895 38453 32293 33135 39928 24841 24561 17484|24964 22935 43076 19519 45535 38274|1511 21137 5194 4920 44275 33497 9934 38313 28204 25072 4793 44473 895 40910 2452 39921 22170 20327 18610 7806 2081 4659 47766 17484 39781|49235 10749 44359 39275 21137 26209 28842 33746 3376|1511 21137 5194 49235 44275 44473 6631 2452 38313 40910 895 39921 22170 20327 10132 24841 39928 9076 21543 12204 3481 47766 17484 36772 18610 1896|2770 47209 34023 28849 3376 1999 5818 19678 31343 16570 39928 28204 12254 38274|2770 1511 21137 5194 44275 47209 44473 40910 28204 2452 38313 895 25072 22170 39921 33135 3481 18610 18523 35108 36036 16093 48720 21019 1896 26344|34551 3481 35108 24841 1896 26344 38641 21461 35045 49174 9808 1700 12980 6873 39430 14267|4920 4793 21137 44275 38313 1896 38641 26344|34448 22888 10761 9595 48230 32293 22935 45535 4920 44359 10132 1511 21137 22170|13740|37029 48230 2450 49235 5194|9598 4138 10761 42383 13740 24561 22935 4920 44359 10132 1511 44275 2452 895 28204 3481 1896 26344|9934 49235 39928 1511 28204|28985 32175 12916 45066 26790 12144 13740 4793 42265 39928 1511 44275 18610 35108|26790 13740 38453 4920 44359 47766 42265 1511 44275 44473 2452 1896 26344|4920 49235 1511|40604 34126 22474 4138 33198 44375 41220 27104 24964 31343 1511 25072 18610 35108 21461 49174|37119 22888 44375 10895 48230 28993 2450 32293 1511 22170 39921 18610 35108 12980|44912|41220 27104 24964 1511 21461 49174|22474 15143 44375 10761 13740 49235 25072 18610|18200 38383 39561 12144 37220 13740 44359 47766 17484 42265 1511 2770 5194 25072 22170 24841|45066 24852 2770 20995 31343 1999 18479 38562|1511 47766 13176 49235 2770 44275 28204 19048 581 34448 3599 26790 16145 8670 47626 28985|1511 21137 24852 4920 2452 2770 3599 22474 34448 31869 49235 25072 6552 8056 28204|37220 13176 28204 42265 7751 48416 34942 21603 21108 3376 44359 39558 27104 39984 22935 39921 34635 42701|24891 1511 13176 47766 48679 44375 21108 22825 4920 39928|2770 13176 28204 4920 47766 26790 2452 44275 44375 25072 5194 39921 18610 42265 22170 21137 10761 26344 1896 10132 45840 44116 4138 9092 33198 4957 31869 43442 3481 23543 6631 21019 31040 41787 36772 27104 46720 27663 3376 32906 5876 2447|49235 47766 13176 21137 40706 20536 44035 43789 4056 10948|1511 11736 8988 30391 44359 19677 19057 21137 44422 21938 2295 13212|49235 37067 8277 47209 5077 33746 8006 45007 23400 5818 19677 22504 23543 6647 39928 31263 13740 39561 25072 18523 37220|9934 5077 1511 37067 47766 45066 2452 39928 24841 26790 42265 31040 27104 1999|18696 28849 14267 13176 47766 2770 1511 49235 40516 40706 30391 3989|13176 19677 28204 47209 48679 15040 784 10163 44422|9934 1511 24891 49235 31040 13176 32175 21603|47766 28204 2770 13740|1511 13176 47766 2770 49235 13740 24891 2452 9934 42265 4920 44275 10132 18523 23543 4957 11736 22888 6873 10895 12980 2450 4163 48325 42383 2651|21290 13380 1511 13176 34448'

prod = range(0, 49700)
product_size = len(prod)

basket_seq = prod_seq.split('|')
iteration = len(basket_seq)
train_x = []
label = []
count = 0
for i in range(0,iteration-1):
    tfrom = list(map(int,basket_seq[i].split(' ')))
    tto = list(map(int,basket_seq[i+1].split(' ')))
    for j in range(len(tfrom)):
        for item in tto:
            train_x.append(tfrom[j])
            label.append(item)

train_x = np.asarray(train_x)

label = np.asarray(label)
len(label)
label_x= np.reshape(label,(len(label),1))


data_index=0
def generate_batch(batch_size):
    global data_index
    if (data_index + batch_size) > len(train_x):
        index = len(train_x)- batch_size
        x= train_x[index: index+batch_size]
        y = label_x[index: index+batch_size,]
        data_index = index + batch_size
    else:
        x= train_x[data_index: data_index+batch_size]
        y = label_x[data_index: data_index+batch_size,]
        data_index += batch_size
    return x,y

generate_batch(128)
import math
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
       # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([product_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([product_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([product_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=product_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()


num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = valid_examples[i]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = nearest[k]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
