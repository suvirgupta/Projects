import os


os.environ["SPARK_HOME"] = "/usr/lib/spark/"
os.environ["PYSPARK_PYTHON"]= "/home/cloudera/anaconda2/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"]= "/home/cloudera/anaconda2/bin/python"
os.environ["SPARK_YARN_USER_ENV"]= "/home/cloudera/anaconda2/bin/python"

from pyspark import SparkConf , SparkContext
conf = (SparkConf().setMaster("local[*]").setAppName("test"))
sc = SparkContext(conf =conf)
# https://stackoverflow.com/questions/34685905/how-to-link-pycharm-with-pyspark  ## for the python on pyspark

# export PYSPARK_PYTHON=/usr/local/bin/python2.7
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python2.7
# export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=/usr/local/bin/python2.7"

# conda install -c blaze py4j


orders = sc.textFile("file:/home/cloudera/Desktop/orders/orders.csv")
orders.take(10)

orders = sc.textFile("/user/cloudera/kaggle/orders/orders.csv").filter(lambda x: 'order_id' not in x).map(lambda x : x.split(','))

orders_prior = sc.textFile("/user/cloudera/kaggle/orders_Prior/order_products__prior.csv").filter(lambda x: 'order_id' not in x).\
    map(lambda x : x.split(','))

orders_train = sc.textFile("/user/cloudera/kaggle/orders_Train/order_products__train.csv").filter(lambda x: 'order_id' not in x).\
    map(lambda x : x.split(','))

ord_train = orders_train.map(lambda x : (int(x[0]),x[1]))
ord_train.take(2)
# u'order_id', u'product_id', u'add_to_cart_order', u'reordered'

ord=orders.filter(lambda x : x[2]!="train").map(lambda x : (int(x[0]),(x[1],x[2],x[3],x[4],x[5],x[6])))
ord.sortByKey().take(10)

or_train= ord_train.groupByKey().mapValues(list)

ord.join(or_train).take(10)

or_train.mapValues(lambda x: ' '.join(x))

or_train.mapValues(lambda x: ' '.join(x)).map(lambda x: ','.join(x)).saveAsTextFile("")

or_pr_train = sc.textFile("/user/cloudera/kaggle/orders_Train/order_train").map(lambda x : x.split(','))
or_cp_tr = sc.textFile("/user/cloudera/kaggle/order").map(lambda x : x.split(','))
# adding elememts to a tuple
# a = ('2','3','4')
# b = 'z'
# new = a + (b,)
def nullparse(x):
    if x =='':
        x=0
    else:
        float(x)
    return x
## Saving joined data sets of orders train and orders_product_train by order_id
or_cp_tr.map(lambda x : (int(x[0]),(x[1],x[2],x[3],x[4],x[5],nullparse(x[6])))).join(or_pr_train.map(lambda x : (int(x[0]),x[1]) ))\
    .mapValues(lambda x : (x[0]+(x[1],))).map(lambda x: (x[0],)+x[1] ).saveAsTextFile("/user/cloudera/kaggle/order_prod_join_train")


#Sample output
[(262240, u'90958', u'train', u'83', u'3', u'18', u'2.0', u'21137 30994 11777 12384 46847 35495 35561'),
 (2704240, u'44324', u'train', u'6', u'6', u'19', u'3.0', u'651 26940 44024 10'),
 (3146000, u'20884', u'train', u'10', u'0', u'15', u'30.0', u'32403 16797 1700 11943 38739 5183 37725 46881 38415 36360 15364 13755 33075 8767 37634 8366 32813 34739 20590 35289 1313 43879 8228 43632 33894 19539 45889 44843 43857 18382 1139 44598 18394 26312'),
 (2540160, u'118140', u'train', u'7', u'0', u'10', u'27.0', u'5750 39275 24852 2962 27086 43552 30233 41714 25506 13873 14227 38730 35003 33568 22849 9020 6403 33100 35761 19340 15359 2394 41581 47626 41273'),
 (2884080, u'169454', u'train', u'13', u'0', u'07', u'30.0', u'39275'),
 (445200, u'46432', u'train', u'5', u'3', u'22', u'4.0', u'43712 39628 28560 31681'),
 (87520, u'73092', u'train', u'6', u'3', u'16', u'23.0', u'13176 8803 8013 46129 39863'),
 (263040, u'142576', u'train', u'5', u'5', u'10', u'10.0', u'13176 21903 33395'),
 (3408800, u'173224', u'train', u'11', u'3', u'06', u'11.0', u'32505 28204 13176 44910 27344 19816 47626 35004 23801 40386 46979 5450'),
 (1243920, u'1393', u'train', u'6', u'0', u'12', u'30.0', u'28934 20995 6347 33198 25466 24838 33810')]