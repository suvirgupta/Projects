

###################################################################################################################################################

#  Refering to paper on Factorizing Personalized Markov Chains for Next-Basket Recommendation
#  BY Steffen Rendle, Christoph Freudenthaler Lars Schmidt-Thieme
### I am only trying to implement the generalized Markov chain model not the personalized one but it also has memory issue
#### reference link
### https://www.ismll.uni-hildesheim.de/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf
####################################################################################################################################################

import os

os.environ["SPARK_HOME"]= '/usr/lib/spark'

os.environ["PYSPARK_PYTHON"]= "/home/cloudera/anaconda2/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"]= "/home/cloudera/anaconda2/bin/python"
os.environ["SPARK_YARN_USER_ENV"]= "/home/cloudera/anaconda2/bin/python"
from pyspark import SparkConf, SparkContext

from pyspark.sql import SQLContext, Row
conf = (SparkConf().setMaster("local[*]").setAppName("test").set("spark.executor.memory","4g").set("spark.driver.memory", "5g"))

sc = SparkContext(conf = conf)

### In preprocessing step we have complied and grouped all transitions based that are user specific
### taking just one user specific transition of purchases to build a container function that can be passed to the spark
prod_seq= '1511 18523 9934 47209 44473 4920 13740 40910 21290 21137 44275 49235 28204 4793 33497 10895 41950 25072 48230 28993 5194 38313 42383 15700 5077 39921 2450 35108 31263 43789 21903 41544 18610 49174 21461 30336 39928 37220 17979|1511 21137 5194 2770 44275 4920 13740 47209 6631 38313 25072 4793 28204 21290 895 38453 32293 33135 39928 24841 24561 17484|24964 22935 43076 19519 45535 38274|1511 21137 5194 4920 44275 33497 9934 38313 28204 25072 4793 44473 895 40910 2452 39921 22170 20327 18610 7806 2081 4659 47766 17484 39781|49235 10749 44359 39275 21137 26209 28842 33746 3376|1511 21137 5194 49235 44275 44473 6631 2452 38313 40910 895 39921 22170 20327 10132 24841 39928 9076 21543 12204 3481 47766 17484 36772 18610 1896|2770 47209 34023 28849 3376 1999 5818 19678 31343 16570 39928 28204 12254 38274|2770 1511 21137 5194 44275 47209 44473 40910 28204 2452 38313 895 25072 22170 39921 33135 3481 18610 18523 35108 36036 16093 48720 21019 1896 26344|34551 3481 35108 24841 1896 26344 38641 21461 35045 49174 9808 1700 12980 6873 39430 14267|4920 4793 21137 44275 38313 1896 38641 26344|34448 22888 10761 9595 48230 32293 22935 45535 4920 44359 10132 1511 21137 22170|13740|37029 48230 2450 49235 5194|9598 4138 10761 42383 13740 24561 22935 4920 44359 10132 1511 44275 2452 895 28204 3481 1896 26344|9934 49235 39928 1511 28204|28985 32175 12916 45066 26790 12144 13740 4793 42265 39928 1511 44275 18610 35108|26790 13740 38453 4920 44359 47766 42265 1511 44275 44473 2452 1896 26344|4920 49235 1511|40604 34126 22474 4138 33198 44375 41220 27104 24964 31343 1511 25072 18610 35108 21461 49174|37119 22888 44375 10895 48230 28993 2450 32293 1511 22170 39921 18610 35108 12980|44912|41220 27104 24964 1511 21461 49174|22474 15143 44375 10761 13740 49235 25072 18610|18200 38383 39561 12144 37220 13740 44359 47766 17484 42265 1511 2770 5194 25072 22170 24841|45066 24852 2770 20995 31343 1999 18479 38562|1511 47766 13176 49235 2770 44275 28204 19048 581 34448 3599 26790 16145 8670 47626 28985|1511 21137 24852 4920 2452 2770 3599 22474 34448 31869 49235 25072 6552 8056 28204|37220 13176 28204 42265 7751 48416 34942 21603 21108 3376 44359 39558 27104 39984 22935 39921 34635 42701|24891 1511 13176 47766 48679 44375 21108 22825 4920 39928|2770 13176 28204 4920 47766 26790 2452 44275 44375 25072 5194 39921 18610 42265 22170 21137 10761 26344 1896 10132 45840 44116 4138 9092 33198 4957 31869 43442 3481 23543 6631 21019 31040 41787 36772 27104 46720 27663 3376 32906 5876 2447|49235 47766 13176 21137 40706 20536 44035 43789 4056 10948|1511 11736 8988 30391 44359 19677 19057 21137 44422 21938 2295 13212|49235 37067 8277 47209 5077 33746 8006 45007 23400 5818 19677 22504 23543 6647 39928 31263 13740 39561 25072 18523 37220|9934 5077 1511 37067 47766 45066 2452 39928 24841 26790 42265 31040 27104 1999|18696 28849 14267 13176 47766 2770 1511 49235 40516 40706 30391 3989|13176 19677 28204 47209 48679 15040 784 10163 44422|9934 1511 24891 49235 31040 13176 32175 21603|47766 28204 2770 13740|1511 13176 47766 2770 49235 13740 24891 2452 9934 42265 4920 44275 10132 18523 23543 4957 11736 22888 6873 10895 12980 2450 4163 48325 42383 2651|21290 13380 1511 13176 34448'

product = sc.textFile("file:///home/cloudera/Desktop/kaggle/products.csv",3)
prod_sorted = product.filter(lambda x : 'product_id' not in x).map(lambda x : x.split(',')).map(lambda x : int(x[0])).sortBy(lambda x : x)

prod = prod_sorted.collect()
len(prod)



t = [0,0]
list1= []
for _ in range(len(prod)):
    list1.append(t)

from collections import defaultdict
user_tra = defaultdict(list1)
import numpy as np

##############################################################################################################
# The problem here is the I am not able to assign a value to dictionary selectively based on key whenever
# I try to update list value in dictionary specific to a key all keys are updated with same values
# def trans_matrix(prod_seq):
###############################################################################################################

basket_seq = prod_seq.split('|')
iteration = len(basket_seq)
for j in range(iteration-1):
    trans_from = basket_seq[j]
    trans_to = basket_seq[j+1]
    tfrom = map(int,trans_from.split(' '))
    tto = map(int,trans_to.split(' '))

    for item in tfrom:
        ### problem here is in each iteration the default value for all keys is updated from [(0,0),(0,0),....] to item_list
        item_list = user_tra[item]

        print item_list[1000:1200]
        print item

        for i in range(len(prod)):
            if i+1 in tto:
               x =  item_list[i]

               x[0] = x[0] +1
               x[1] = x[1] +1
               item_list[i] = x
            else:
                x = item_list[i]

                x[0] = x[0]
                x[1] = x[1] + 1
                item_list[i] = x
    user_tra[item]= item_list
    del item_list
    print user_tra[item][1000:1200]





###########################################################################################################################
#### Since above code does not work with the dictionary of list of list due to some unkown assignment by reference error
## issue: https://stackoverflow.com/questions/45537605/transition-count-for-user-product-purchases-in-online-retail
#### next we might use multidimentional numpy array instead of dictionary of list of list
#### this implementation has its own issue of the memory constrains as there are around 50000 products
#### user transition matrix would be of the dimension 50000*50000*2 if each point in matrix takes 4bytes it comes to more than 12gb of space in memory
#### which is not possible on personal computer even 16 gb of ram
#### so I am taking it into consideration small and manageable problem with products in range 50
#### to get the core of algorithm and then it can be scaled for the larger data set
##############################################################################################################################################

import numpy as np

user_purchase = '3 4 12 23 45 41 25|4 5 12 17 19 25 46 3|39 12 3 23 50 24 35 13|42 34 17 19 46'
prod = range(0, 50)
user_tran = np.zeros((50,50,2))
basket_seq = user_purchase.split('|')
iteration = len(basket_seq)
for j in range(iteration-1):
    trans_from = basket_seq[j]
    trans_to = basket_seq[j+1]
    tfrom = map(int,trans_from.split(' '))
    #tfrom = [x-1 for x in tfrom]
    tto = map(int,trans_to.split(' '))
    #tto = [x - 1 for x in tto]
    for item in tfrom:
        item_list = user_tran[item-1, :, :]
        for i in range(len(prod)):
            if i + 1 in tto:
                temp = item_list[i, :]
                item_list[i, :] = np.array([temp[0] + 1, temp[1] + 1])
            else:
                temp = item_list[i, :]
                item_list[i, :] = np.array([temp[0], temp[1] + 1])
        user_tran[item-1, :, :] = item_list
print user_tran[3, :, :] ## print elements of transition of product 4 in all transitions

# output for the 4rd item look like this, transition from product 4 to 1,2,3,4.....50
# [0. 2.] implies two transitions occured from 4 no was to the product 1
# [[ 0.  2.][ 0.  2.][ 2.  2.][ 1.  2.][ 1.  2.][ 0.  2.][ 0.  2.][ 0.  2.][ 0.  2.][ 0.  2.][ 0.  2.][ 2.  2.][ 1.  2.]..........[ 0.  2.][ 0.  2.][ 1.  2.][ 0.  2.][ 0.  2.][ 0.  2.][ 1.  2.]]

# now we can compute the transition probabilities by dividing the first element of the two element array with the denominator
#
#### Computing transition probabilities
user_tran_pb_matrix = np.zeros((50,50))

for product in range(len(prod)):
    prod_trans = user_tran[product,:,:]
    for trans in range(len(prod_trans)):
        val = prod_trans[trans]
        try:
            user_tran_pb_matrix[product,trans]= float(val[0]/val[1])
        except :
            user_tran_pb_matrix[product, trans]= 0.0



def div0( a ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a[0], a[1] )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c
n= user_tran[26,:,:]
p = n[2]

user_tran_pb_matrix[2,:]
# array([ 0.        ,  0.        ,  0.66666667,  0.33333333,  0.33333333,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.66666667,  0.33333333,  0.        ,  0.        ,
#         0.        ,  0.66666667,  0.        ,  0.66666667,  0.        ,
#         0.        ,  0.        ,  0.33333333,  0.33333333,  0.33333333,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.33333333,  0.33333333,
#         0.        ,  0.        ,  0.        ,  0.33333333,  0.        ,
#         0.        ,  0.33333333,  0.        ,  0.        ,  0.        ,
#         0.66666667,  0.        ,  0.        ,  0.        ,  0.33333333])
#

### Convert matrix to pandas Dataframe

import pandas as pd
pd.DataFrame(user_tran_pb_matrix)
### Here nan indicates product was never purchased by user in entire history
### 0 implies product was pruchased at least once but didnot transitioned to other product
### Tble gives the probability of transition from 1 product to other as array starts from 0 add 1 to get the correct index
### eg probability of transition from 3-3 is 0.666667
     # 0    1         2         3         4    5    6    7    8    9     ...     \
# 0   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 1   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 2   0.0  0.0  0.666667  0.333333  0.333333  0.0  0.0  0.0  0.0  0.0    ...
# 3   0.0  0.0  1.000000  0.500000  0.500000  0.0  0.0  0.0  0.0  0.0    ...
# 4   0.0  0.0  1.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 5   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 6   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 7   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 8   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 9   NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 10  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 11  0.0  0.0  0.666667  0.333333  0.333333  0.0  0.0  0.0  0.0  0.0    ...
# 12  0.0  0.0  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 13  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 14  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 15  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 16  0.0  0.0  1.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 17  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 18  0.0  0.0  1.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 19  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 20  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 21  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 22  0.0  0.0  0.500000  0.500000  0.500000  0.0  0.0  0.0  0.0  0.0    ...
# 23  0.0  0.0  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 24  0.0  0.0  1.000000  0.500000  0.500000  0.0  0.0  0.0  0.0  0.0    ...
# 25  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 26  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 27  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 28  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 29  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 30  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 31  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 32  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 33  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 34  0.0  0.0  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 35  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 36  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 37  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 38  0.0  0.0  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0    ...
# 39  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 40  0.0  0.0  1.000000  1.000000  1.000000  0.0  0.0  0.0  0.0  0.0    ...
# 41  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...
# 42  NaN  NaN       NaN       NaN       NaN  NaN  NaN  NaN  NaN  NaN    ...


#      40        41   42   43   44        45   46   47   48        49
# 0   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 1   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 2   0.0  0.333333  0.0  0.0  0.0  0.666667  0.0  0.0  0.0  0.333333
# 3   0.0  0.000000  0.0  0.0  0.0  0.500000  0.0  0.0  0.0  0.500000
# 4   0.0  0.000000  0.0  0.0  0.0  0.000000  0.0  0.0  0.0  1.000000
# 5   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 6   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 7   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 8   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 9   NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 10  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 11  0.0  0.333333  0.0  0.0  0.0  0.666667  0.0  0.0  0.0  0.333333
# 12  0.0  1.000000  0.0  0.0  0.0  1.000000  0.0  0.0  0.0  0.000000
# 13  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 14  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 15  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 16  0.0  0.000000  0.0  0.0  0.0  0.000000  0.0  0.0  0.0  1.000000
# 17  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 18  0.0  0.000000  0.0  0.0  0.0  0.000000  0.0  0.0  0.0  1.000000
# 19  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 20  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 21  NaN       NaN  NaN  NaN  NaN       NaN  NaN  NaN  NaN       NaN
# 22  0.0  0.500000  0.0  0.0  0.0  1.000000  0.0  0.0  0.0  0.000000
# 23  0.0  1.000000  0.0  0.0  0.0  1.000000  0.0  0.0  0.0  0.000000
# 24  0.0  0.000000  0.0  0.0  0.0  0.500000  0.0  0.0  0.0  0.500000

##########################################################################################################
#### Now we van appy matrix factorization on the 'user_tran_pb_matrix' to get the recommendations
#### Generic matrix factorization is already been shown inthe Model.Prediction Step
##########################################################################################################

