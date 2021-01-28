{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import sys\n",
    "import matplotlib.pyplot as plt\n",
    "import re\n",
    "import numpy as np\n",
    "from scipy import stats\n",
    "import statsmodels.api as sm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Loading the data from the CSV file into a dataframe \n",
    "df1= pd.read_csv('dog_rates_tweets.csv')\n",
    "#df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = df1.text.str.extract(r'(\\d+(\\.\\d+)?)/10')\n",
    "#Dropping all rows containing Nan\n",
    "data.dropna(subset=[0], axis = 0 , inplace= True)\n",
    "#Converting object datatype of the column to float datatype\n",
    "data=pd.to_numeric(data[0])\n",
    "#data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Cleaning the data further\n",
    "mask=data<25\n",
    "data=data[mask]\n",
    "#data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1['Ratings']=data\n",
    "#To display the data without Nan\n",
    "df1 = df1.dropna()\n",
    "#df1\n",
    "#df1.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Converting created_at column to a datetime value\n",
    "df1['created_at']= pd.to_datetime(df1['created_at'],format=\"%Y-%m-%d %H:%M:%S\") \n",
    "#df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>created_at</th>\n",
       "      <th>text</th>\n",
       "      <th>Ratings</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>994363623421153280</td>\n",
       "      <td>2018-05-09 23:48:56</td>\n",
       "      <td>This is Louie. He has misplaced his Cheerio. W...</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1.525910e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>993889039714578432</td>\n",
       "      <td>2018-05-08 16:23:07</td>\n",
       "      <td>This is Manny. He hasn’t seen your croissant. ...</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.525797e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>993629544463642624</td>\n",
       "      <td>2018-05-07 23:11:58</td>\n",
       "      <td>This is Libby. She leap. 14/10\\n(IG: libbythef...</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1.525735e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>992198572664860672</td>\n",
       "      <td>2018-05-04 00:25:48</td>\n",
       "      <td>This is Rosie. She thought Coachella was this ...</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.525394e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>991744041351090177</td>\n",
       "      <td>2018-05-02 18:19:39</td>\n",
       "      <td>This is Riley. He’ll be your chauffeur this ev...</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.525285e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11631</th>\n",
       "      <td>1096203765189726208</td>\n",
       "      <td>2019-02-15 00:25:18</td>\n",
       "      <td>honorary 15/10 for Oppy, the very good space r...</td>\n",
       "      <td>15.0</td>\n",
       "      <td>1.550190e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11638</th>\n",
       "      <td>1095730341828915200</td>\n",
       "      <td>2019-02-13 17:04:05</td>\n",
       "      <td>This is George. He doesn’t chew socks. He just...</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1.550077e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11665</th>\n",
       "      <td>1093636946046242817</td>\n",
       "      <td>2019-02-07 22:25:41</td>\n",
       "      <td>@Panthers @Proud_KCS 13/10 easy</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.549578e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11666</th>\n",
       "      <td>1093636812818472960</td>\n",
       "      <td>2019-02-07 22:25:09</td>\n",
       "      <td>RT @GeekandSundry: .@Dog_Rates Announces a New...</td>\n",
       "      <td>13.0</td>\n",
       "      <td>1.549578e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11672</th>\n",
       "      <td>1093315910369107968</td>\n",
       "      <td>2019-02-07 01:10:00</td>\n",
       "      <td>This is Missy. These are her best angles. She ...</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1.549502e+09</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1871 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                        id          created_at  \\\n",
       "2       994363623421153280 2018-05-09 23:48:56   \n",
       "7       993889039714578432 2018-05-08 16:23:07   \n",
       "8       993629544463642624 2018-05-07 23:11:58   \n",
       "24      992198572664860672 2018-05-04 00:25:48   \n",
       "30      991744041351090177 2018-05-02 18:19:39   \n",
       "...                    ...                 ...   \n",
       "11631  1096203765189726208 2019-02-15 00:25:18   \n",
       "11638  1095730341828915200 2019-02-13 17:04:05   \n",
       "11665  1093636946046242817 2019-02-07 22:25:41   \n",
       "11666  1093636812818472960 2019-02-07 22:25:09   \n",
       "11672  1093315910369107968 2019-02-07 01:10:00   \n",
       "\n",
       "                                                    text  Ratings  \\\n",
       "2      This is Louie. He has misplaced his Cheerio. W...     14.0   \n",
       "7      This is Manny. He hasn’t seen your croissant. ...     13.0   \n",
       "8      This is Libby. She leap. 14/10\\n(IG: libbythef...     14.0   \n",
       "24     This is Rosie. She thought Coachella was this ...     13.0   \n",
       "30     This is Riley. He’ll be your chauffeur this ev...     13.0   \n",
       "...                                                  ...      ...   \n",
       "11631  honorary 15/10 for Oppy, the very good space r...     15.0   \n",
       "11638  This is George. He doesn’t chew socks. He just...     14.0   \n",
       "11665                    @Panthers @Proud_KCS 13/10 easy     13.0   \n",
       "11666  RT @GeekandSundry: .@Dog_Rates Announces a New...     13.0   \n",
       "11672  This is Missy. These are her best angles. She ...     12.0   \n",
       "\n",
       "          timestamp  \n",
       "2      1.525910e+09  \n",
       "7      1.525797e+09  \n",
       "8      1.525735e+09  \n",
       "24     1.525394e+09  \n",
       "30     1.525285e+09  \n",
       "...             ...  \n",
       "11631  1.550190e+09  \n",
       "11638  1.550077e+09  \n",
       "11665  1.549578e+09  \n",
       "11666  1.549578e+09  \n",
       "11672  1.549502e+09  \n",
       "\n",
       "[1871 rows x 5 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def to_timestamp(t):\n",
    "    return t.timestamp()\n",
    "df1['timestamp'] = df1['created_at'].apply(to_timestamp)\n",
    "df1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2.2983031752244575e-08, -22.445058882763483)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fit = stats.linregress(df1['timestamp'], df1['Ratings'])\n",
    "fit.slope, fit.intercept"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.xticks(rotation=25)\n",
    "x=df1['created_at'].values\n",
    "y1= df1['Ratings']\n",
    "plt.plot(x,y1, 'b.', alpha=0.5)\n",
    "y2=df1['timestamp']*fit.slope + fit.intercept\n",
    "plt.plot(x, y2, 'r-', linewidth=3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3.793798773108244e-121\n"
     ]
    }
   ],
   "source": [
    "pval=fit.pvalue\n",
    "print(pval)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#From lecture notes\n",
    "y= df1['Ratings']\n",
    "x= df1['timestamp']\n",
    "residuals = y - (fit.slope*x + fit.intercept)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(residuals)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
