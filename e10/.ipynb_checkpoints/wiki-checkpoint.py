{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "assert sys.version_info >= (3, 5) # make sure we have Python 3.5+\n",
    "from pyspark.sql import SparkSession, functions, types\n",
    "\n",
    "# ADD the following lines before doing anything \n",
    "# SET JAVA_HOME=C:\\Program Files\\Android\\Android Studio\\jre\n",
    "# SET HADOOP_HOME=\"C:\\winutils\"  \n",
    "\n",
    "spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()\n",
    "spark.sparkContext.setLogLevel('WARN')\n",
    "\n",
    "assert spark.version >= '2.3' # make sure we have Spark 2.3+\n",
    "\n",
    "wikipedia_schema = types.StructType([\n",
    "    types.StructField('lang', types.StringType()),\n",
    "    types.StructField('title', types.StringType()),\n",
    "    types.StructField('times',types.LongType()),\n",
    "    types.StructField('bytes',types.LongType()),\n",
    "])\n",
    "\n",
    "def main(inputs, output):\n",
    "    # main logic starts here\n",
    "    weather = spark.read.csv(inputs, schema=wikipedia_schema, sep=\" \").withColumn('filename', functions.input_file_name())\n",
    "   \n",
    "    weather.createOrReplaceTempView(\"temp\")\n",
    "    \n",
    "    sqlDf = spark.sql(\"SELECT regexp_extract(filename, '([0-9]+-[0-9][0-9])') as date, title, times FROM temp WHERE title NOT LIKE 'Special%' AND lang = 'en' AND title != 'Main_Page'\")\n",
    "    sqlDf = sqlDf.cache()\n",
    "    max_count = sqlDf.groupBy('date').max()\n",
    "    \n",
    "    joined_data = sqlDf.join(max_count, on='date')\n",
    "    joined_data = joined_data.filter(\n",
    "        joined_data['times'] == joined_data['max(times)']\n",
    "    )\n",
    "    joined_data = joined_data.select(\n",
    "        'date',\n",
    "        'title',\n",
    "        'max(times)'\n",
    "    )\n",
    "    joined_data = joined_data.orderBy('date', ascending= True)\n",
    "    joined_data.write.csv(output, mode='overwrite')\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    inputs = sys.argv[1]\n",
    "    output = sys.argv[2]\n",
    "    #sc = spark.sparkContext\n",
    "\n",
    "    main(inputs, output)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
