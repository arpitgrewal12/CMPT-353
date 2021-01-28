import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), num_bytes=m.group(2))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    rows = log_lines.map(line_to_row)
    rows = rows.filter(not_none)
    return rows

def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))
    #Group by hostname; get the number of requests and sum of bytes transferred, to form a data point
    counts = logs.groupby('hostname').count().alias('count').cache()
    bytes = logs.groupby('hostname').agg(functions.sum('num_bytes').alias('bytes')).cache()
    data = counts.join(bytes, 'hostname')
    #Produce six values:
    #Add these to get the six sums.
    grouped_data = data.select(data['count'].alias('x'),data['bytes'].alias('y'),(data['count']*data['bytes']).alias('x*y'),(data['count']*data['count']).alias('x^2'),(data['bytes']*data['bytes']).alias('y^2')).cache()
 
    a = grouped_data.agg(functions.sum('x')).first()[0]
    b = grouped_data.agg(functions.sum('y')).first()[0]
    c = grouped_data.agg(functions.sum('x^2')).first()[0]
    d = grouped_data.agg(functions.sum('y^2')).first()[0]
    e = grouped_data.agg(functions.sum('x*y')).first()[0]
    
    #Calculate the final value of r
    
    r = ((data.count()* e)-(a*b))/((math.sqrt((data.count()*c)-(a**2)))*(math.sqrt((data.count()*d)-(b**2))))
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
