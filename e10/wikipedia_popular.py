#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


wiki_schema = types.StructType([
    types.StructField('language', types.StringType(), False),
    types.StructField('name_of_page', types.StringType(), False),
    types.StructField('number_of_views', types.LongType(), False),
    types.StructField('bytes', types.LongType(), False),
])

def format_string(path):
    x=re.search("([0-9]{8}-[0-9]{2})", path).group(1)
    return x


def main(in_directory, output):
    file_data= spark.read.csv(in_directory, schema=wiki_schema, sep=' ').withColumn('filename', functions.input_file_name())
    
    file_data = file_data.filter(file_data['language'] == 'en')
    file_data = file_data.filter(file_data['name_of_page']!='Main_Page')
    file_data = file_data.filter(file_data.name_of_page.startswith('Special:') != True)
    #file_data= file_data.cache()
    #0m16.320s
    path_to_hour = functions.udf(lambda pathname: format_string(pathname), returnType=types.StringType())
    file_data = file_data.withColumn('date', path_to_hour(file_data['filename']))
    #file_data= file_data.cache()
    #14.112s
    max_data = file_data.groupby('date').max('number_of_views')
    file_data = file_data.join(max_data,'date')
    #file_data= file_data.cache()
    #0m13.993s
    file_data = file_data[file_data['number_of_views']==file_data['max(number_of_views)']].select('date','name_of_page','number_of_views')
    file_data= file_data.cache()
    #0m13.475s
    file_data = file_data.sort(file_data['date'],file_data['name_of_page'])
    #file_data= file_data.cache()
    #0m14.098s
    file_data.show()
    file_data = file_data.write.csv(output, mode='overwrite')



if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)

