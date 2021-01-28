import pandas as pd
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

filename = sys.argv[1]
cpu_data = pd.read_csv(filename)

#cpu_data=pd.read_csv('sysinfo.csv')

#Changing the given timestamp value to float
cpu_data['created_at'] = pd.to_datetime(cpu_data['timestamp'], format="%Y-%m-%d %H:%M:%S")

def to_timestamp(dt):
    return dt.timestamp()
cpu_data['timestamp'] = cpu_data['created_at'].apply(to_timestamp)


#LOESS Smoothing
loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp'] , frac=0.05)

#Kalman 
kalman_data=cpu_data[['temperature','cpu_percent','sys_load_1','fan_rpm']]

initial_state=kalman_data.iloc[0]
observation_covariance=np.diag([0.3,0.9,1.8,2.8])** 2
transition_covariance=np.diag([0.2,0.2,0.2,0.2])** 2
transition=[[0.97,0.5,0.2,-0.001],[0.1,0.4,2.2,0],[0,0,0.95,0],[0,0,0,1]]

kf=KalmanFilter(
initial_state_mean=initial_state,
initial_state_covariance=observation_covariance,
observation_covariance=observation_covariance,
transition_covariance=transition_covariance,
transition_matrices=transition
)
kalman_smoothed,_=kf.smooth(kalman_data)

#Plotting the figure
plt.figure(figsize=(12, 4))

plt.plot(cpu_data['created_at'], cpu_data['temperature'], 'b.',alpha=0.5)
plt.plot(cpu_data['created_at'],kalman_smoothed[:,0],'g-')
plt.plot(cpu_data['created_at'], loess_smoothed[:, 1], 'r-')
plt.legend([ 'CPU_Data points','Kalman Smoothing', 'Loess Smoothing'])

plt.savefig('cpu.svg')

