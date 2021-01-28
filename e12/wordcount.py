import sys
from pyspark.sql import SparkSession, functions, types
import string, re

spark = SparkSession.builder.appName('word count').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


def main(in_directory, out_directory):
#   Reading the file
    data = spark.read.text(in_directory)
#   data.show()


#   Split the lines into words with the regular expression below.
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)
#   print(wordbreak)


#   Use the split and explode functions.
    words=data.select(functions.split(data.value, wordbreak).alias('lines')).cache()
#   words.show()
    words = words.select(functions.explode(words['lines']).alias('value'))
#   words.show()


#   Normalize all of the strings to lower-case (so “word” and “Word” are not counted separately.)
    words=words.select(functions.lower(words['value']).alias('word'))
#   words.show()


#   Count the number of times each word occurs.
    words_order=words.groupby('word').count().alias('count').cache()
    
    
#   Sort by decreasing count (i.e. frequent words first) and alphabetically if there's a tie.
    words_order=words_order.orderBy(words_order['count'].desc()).cache()
#   words.show()


#   empty strings being counted: remove them from the output
    word_count=words_order.filter(words_order['word']!= '')
    word_count.show()
    
    
#   Write results as CSV files
    word_count.write.csv(out_directory)

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
