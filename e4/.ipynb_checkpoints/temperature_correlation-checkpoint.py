{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "import sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "file1 = sys.argv[1] \n",
    "#file1 = \"stations.json.gz\"\n",
    "file2 = sys.argv[2] \n",
    "#file2 = \"city_data.csv\"\n",
    "file3 = sys.argv[3] \n",
    "#file3 = 'output.svg'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "stations = pd.read_json(file1, lines=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "stations['avg_tmax']=stations['avg_tmax']/10.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "citiesdata=pd.read_csv(file2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#citiesdata"
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "citiesdata=citiesdata.dropna()\n",
    "#citiesdata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "citiesdata['area']=(citiesdata['area'])/(1000000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "citiesdata = citiesdata[citiesdata['area']<10000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#citiesdata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#The citation below was found on:https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula\n",
    "\n",
    "\n",
    "def distance_between_points(latitude1,longitude1,latitude2,longitude2):\n",
    "    R = 6371 # Radius of earth\n",
    "    Lat = math.radians(latitude2-latitude1) \n",
    "    Lon = math.radians(longitude2-longitude1)\n",
    "    ma=math.sin(Lat/2)\n",
    "    mb=math.sin(Lat/2) \n",
    "    mc=math.cos(math.radians(latitude1))\n",
    "    md=math.cos(math.radians(latitude2))\n",
    "    me=math.sin(Lon/2)\n",
    "    mf= math.sin(Lon/2)\n",
    "    a = ma* mb + mc * md * me * mf\n",
    "    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))\n",
    "    dis = R * c \n",
    "    return dis\n",
    "\n",
    "funct = np.vectorize(distance_between_points)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def distance(citiesdata, stations):\n",
    "    a=citiesdata['latitude']\n",
    "    b=citiesdata['longitude']\n",
    "    c=stations['latitude']\n",
    "    d=stations['longitude']\n",
    "    stations['distance'] = funct(a,b,c,d )\n",
    "    return stations['distance']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def best_tmax(citiesdata, stations):\n",
    "    stations['distance'] = distance(citiesdata, stations)\n",
    "    return stations['avg_tmax'][stations['distance'].idxmin()]\n",
    "#stations\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "citiesdata['population density'] = citiesdata['population']/citiesdata['area']\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "citiesdata['best_tmax'] = citiesdata.apply(best_tmax, 1, stations=stations)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
    "plt.plot(citiesdata['best_tmax'], citiesdata['population density'], 'b.')\n",
    "plt.title('Relationship between Temperature and Population Density')\n",
    "plt.xlabel('Avg Max Temperature (°C)')\n",
    "plt.ylabel('Population Density (people/km²')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.savefig(file3)"
   ]
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
