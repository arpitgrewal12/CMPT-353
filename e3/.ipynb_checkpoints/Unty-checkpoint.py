{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import sys\n",
    "import xml.etree.cElementTree as et\n",
    "from math import radians, cos, sin, asin, sqrt\n",
    "import numpy as np\n",
    "from pykalman import KalmanFilter\n",
    "\n",
    "def smooth(kalman_data):\n",
    "    initial_state = kalman_data.iloc[0]\n",
    "    observation_covariance = np.diag([0.000017, .000017]) ** 2 \n",
    "    transition_covariance = np.diag([0.00001, 0.00001]) ** 2 \n",
    "    kf = KalmanFilter(\n",
    "        initial_state_mean=initial_state,\n",
    "        initial_state_covariance=observation_covariance,\n",
    "        observation_covariance=observation_covariance,\n",
    "        transition_covariance=transition_covariance,\n",
    "    )\n",
    "    kalman_smoothed, state_cov = kf.smooth(kalman_data)\n",
    "\n",
    "    return pd.DataFrame(kalman_smoothed, columns=['lat', 'lon'])\n",
    "\n",
    "## taken from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points\n",
    "def hav_distance(lat1, lon1, lat2, lon2):\n",
    "    \"\"\"\n",
    "    Calculate the great circle distance between two points \n",
    "    on the earth (specified in decimal degrees)\n",
    "    \"\"\"\n",
    "    # convert decimal degrees to radians \n",
    "    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])\n",
    "\n",
    "    # haversine formula \n",
    "    dlon = lon2 - lon1 \n",
    "    dlat = lat2 - lat1 \n",
    "    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2\n",
    "    c = 2 * asin(sqrt(a)) \n",
    "    r = 6371000 # Radius of earth in meters. Use 3956 for miles\n",
    "    return c * r \n",
    "\n",
    "def distance(df):\n",
    "    df['prev_lat'] = df.shift()['lat']\n",
    "    df['prev_lon'] = df.shift()['lon']\n",
    "    # df = df.iloc[1:]\n",
    "    df['distances'] = df.apply(lambda x: hav_distance(x['lat'], x['lon'], x['prev_lat'], x['prev_lon']), axis=1)\n",
    "    total = df['distances'].sum()\n",
    "    return total\n",
    "\n",
    "def get_data(xmlFile):\n",
    "    parsedXML = et.parse( xmlFile)\n",
    "    df =pd.DataFrame(columns=['lat', 'lon'], dtype=np.float)\n",
    "    for item in parsedXML.iter('{http://www.topografix.com/GPX/1/0}trkpt'):\n",
    "        df = df.append(item.attrib, ignore_index=True)\n",
    "    return df.astype('float64')\n",
    "\n",
    "\n",
    "def output_gpx(points, output_filename):\n",
    "    \"\"\"\n",
    "    Output a GPX file with latitude and longitude from the points DataFrame.\n",
    "    \"\"\"\n",
    "    from xml.dom.minidom import getDOMImplementation\n",
    "    def append_trkpt(pt, trkseg, doc):\n",
    "        trkpt = doc.createElement('trkpt')\n",
    "        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))\n",
    "        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))\n",
    "        trkseg.appendChild(trkpt)\n",
    "    \n",
    "    doc = getDOMImplementation().createDocument(None, 'gpx', None)\n",
    "    trk = doc.createElement('trk')\n",
    "    doc.documentElement.appendChild(trk)\n",
    "    trkseg = doc.createElement('trkseg')\n",
    "    trk.appendChild(trkseg)\n",
    "    \n",
    "    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)\n",
    "    \n",
    "    with open(output_filename, 'w') as fh:\n",
    "        doc.writexml(fh, indent=' ')\n",
    "\n",
    "\n",
    "def main():\n",
    "    points = get_data(sys.argv[1])\n",
    "    print('Unfiltered distance: %0.2f' % (distance(points),))\n",
    "\n",
    "    points = get_data(sys.argv[1])\n",
    "    smoothed_points = smooth(points)\n",
    "    print('Filtered distance: %0.2f' % (distance(smoothed_points),))\n",
    "    output_gpx(smoothed_points, 'out.gpx')\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
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
