import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
   comments = spark.read.json(in_directory, schema=comments_schema).cache()
   #Calculate the average score for each subreddit, as before.
   groups = comments.groupBy('subreddit')
   averages = groups.agg(functions.avg(comments.score).alias("avgscore")).cache()
   
   #Exclude any subreddits with average score ≤0.
   averages= averages.filter(averages.avgscore>0)

   #Join the average score to the collection of all comments. Divide to get the relative score.
   averages = averages.join(functions.broadcast(comments), 'subreddit','inner')
  #averages = averages.join(comments, 'subreddit')
   averages = averages.withColumn('relative_score', averages.score/averages.avgscore)
   averages = averages.groupBy('subreddit').agg(functions.max(averages.relative_score).alias("rel_score"))
   
   #Determine the max relative score for each subreddit.
   scores= groups.agg(functions.max(comments.score).alias("score")).cache()
   scores = scores.join(functions.broadcast(comments), ['score','subreddit'],'inner')
   #scores = scores.join(comments, ['score','subreddit'])

   #Join again to get the best comment on each subreddit: we need this step to get the author.
   best_score_author = scores.join(functions.broadcast(averages), 'subreddit','inner')
   #best_score_author = scores.join(averages, 'subreddit')
   best_score_author = best_score_author.select('subreddit', 'author', 'rel_score')
   
   #best_score_author.show()
   best_score_author.write.json(out_directory, mode='overwrite')


   
if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
