from xml.dom.minidom import parse, parseString
import pandas as pd
import numpy as np
import sys
import math
from pykalman import KalmanFilter

#Reading the xml file
def read_xml(xmlfile):
    d = parse(xmlfile).getElementsByTagName('trkpt')
    df = pd.DataFrame(columns=['lat', 'lon'])
    #Loop for iterating through the XML file
    for i in range(len(d)):
        df.loc[i]=float(d[i].attributes['lat'].value), float(d[i].attributes['lon'].value)
    return df

def smooth(kalman_data):
    initial_value_guess = kalman_data.iloc[0]
    observation_covariance = np.diag([0.000017, .000017]) ** 2
    transition_covariance = np.diag([0.00001, 0.00001]) ** 2
    transition_matrix = [[1, 0], [0, 1]]
    kf = KalmanFilter(initial_state_mean=initial_value_guess,
                initial_state_covariance=observation_covariance,
                observation_covariance=observation_covariance,
                transition_covariance=transition_covariance,
                transition_matrices=transition_matrix)
    kalman_smoothed, _ = kf.smooth(kalman_data)
    result = pd.DataFrame(kalman_smoothed,columns=['lat','lon'])
    return result

#Some reference was taken from the stack overflow link given in the assignment
#http://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance_between_points(latitude1,longitude1,latitude2,longitude2):
    R = 6371 # Radius of earth
    Lat = math.radians(latitude2-latitude1) 
    Lon = math.radians(longitude2-longitude1)
    ma=math.sin(Lat/2)
    mb=math.sin(Lat/2) 
    mc=math.cos(math.radians(latitude1))
    md=math.cos(math.radians(latitude2))
    me=math.sin(Lon/2)
    mf= math.sin(Lon/2)
    a = ma* mb + mc * md * me * mf
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dis = R * c 
    return dis

dis = np.vectorize(distance_between_points)

def distance(dta):
    a=dta['lat']
    b=dta['lon']
    c=dta['lat'].shift(1)
    d=dta['lon'].shift(1)
    dta['dist'] = dis(a,b,c,d)
    d_sum = dta['dist'].sum()*1000
    del dta['dist']
    return d_sum 

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def main():
    points =read_xml(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')
     
if __name__ == '__main__':
    main()


