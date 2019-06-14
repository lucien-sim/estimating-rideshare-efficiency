import os
data_path = './data'
default_crs = 'epsg:4326'
original_csv = os.path.join(data_path,'yellow_tripdata_2016-06.csv')
processed_pkl = os.path.join(data_path,'manhattan_rides.pkl')
saved_rides_pkl = os.path.join(data_path,'saved_rides.pkl')