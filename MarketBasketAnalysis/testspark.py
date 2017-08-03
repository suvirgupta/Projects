import os

os.environ["SPARK_HOME"]= '/usr/lib/spark'

os.environ["PYSPARK_PYTHON"]= "/home/cloudera/anaconda2/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"]= "/home/cloudera/anaconda2/bin/python"
os.environ["SPARK_YARN_USER_ENV"]= "/home/cloudera/anaconda2/bin/python"
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

conf = (SparkConf().setMaster("local[*]").setAppName("test").set("spark.executor.memory","4g").set("spark.driver.memory", "5g"))

sc = SparkContext(conf = conf)

joined_prior =  sc.textFile("/user/cloudera/kaggle/order_product_prior_joined")
inter_join=joined_prior.map(lambda x : x.split(',')).map(lambda x : (int(x[1]), (x[0], x[3],x[4],x[6],x[7] ))).\
    combineByKey((lambda x : ((x[0],), (x[1],),(x[2],),(x[3],), (x[4],))),
                 (lambda x,y: (x[0] + (y[0],),x[1] + (y[1],),x[2] + (y[2],),x[3] + (y[3],),x[4] + (y[4],))),
                 (lambda x,y : (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3],x[4]+y[4])))

def tuplesort(a,b,c,d,e):
    combined = list()
    x,y,z,s,t=(),(),(),(),()
    for i in range(len(b)):
        combined.append((a[i],int(b[i]),c[i],d[i],e[i]))
    j= sorted(combined, key= lambda x : x[1])
    for i in range(len(combined)):
        x,y,z,s,t= x + (j[i][0],), y+(j[i][1],), z+ (j[i][2],), s + (j[i][3],), t +(j[i][4],)
    return [x,y,z,s,t]



inter1_join=inter_join.map(lambda x : (x[0], tuplesort(x[1][0], x[1][1], x[1][2],x[1][3],x[1][4])))\
    .map(lambda x : (x[0],' '.join(x[1][0]), ' '.join(x[1][2]),' '.join(x[1][3]),'|'.join(x[1][4]))).\
    map(lambda x : (x[0], x[4]))

inter1_join.persist()

def listcounter(list):
    from collections import Counter
    counts = Counter(list)
    return counts.items()

user_product_matrix = inter1_join.mapValues(lambda x : x.split("|")).mapValues(lambda x : ' '.join(x)).mapValues(lambda x : x.split(' ')).\
    map(lambda x: (x[0],listcounter(x[1]))).map(lambda x : [(x[0],)+ i for i in x[1]]).flatMap(lambda x : x).\
    map(lambda x : (str(x[0]),str(x[1]),str(x[2])) ).map(lambda x : ' '.join(x)).map(lambda x : x.split(' ')).\
    map(lambda x : [int(y) for y in x])

usermap = user_product_matrix.map(lambda x : (x[0],x[1])).collectAsMap()

usermap

x= sc.textFile("/user/cloudera/kaggle/orders").map(lambda x: x.split(','))
test = x.filter(lambda x: x[2]=='test').map(lambda x: x[1]).distinct()
test = test.map(lambda x : int(x))

test_users = test.collect()
test_users
j = user_product_matrix.map(lambda x : (x[0],x[1])).filter(lambda x : x[0] in test_users)

m = j.map(lambda x : (x[1],x[0]))
product = sc.textFile("file:///home/cloudera/Desktop/kaggle/products.csv")

prd = product.filter(lambda x : 'product_id' not in x).map(lambda x: x.split(',')).map(lambda x : (int(x[0]),int(x[2])))

prd.join(m).take(4)

j= test.join(user_product_matrix.map(lambda x : (x[0],x[1])))

j.persist()
j.take(2)
usermap
test.map(lambda x : (x,usermap[x]) ).take(40)
