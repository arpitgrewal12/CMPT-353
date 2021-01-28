
import pandas as pd
import sys 
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]

df1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
#Sorting the values
sorted_df1=df1.sort_values(by=['views'],ascending=False)
#print(df1)

df2=pd.read_csv(filename2, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
#print(df2)

df1['views 2nd hour']=df2['views']
#print(df1)

#Plot1: Distribution of views
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')
plt.plot(sorted_df1['views'].values)

#Plot2: Hourly views
plt.subplot(1,2,2)
x= df1['views'].values
y= df1['views 2nd hour'].values
plt.scatter(x,y)
#loglog scale
plt.yscale('log')
plt.xscale('log')
plt.title('Hourly Correlation')
plt.xlabel('Hour 1 views')
plt.ylabel('Hour 2 views')
#Saving the figure as an image
plt.savefig('wikipedia.png')