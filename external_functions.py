import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import os
import pickle

def load_boroughs(): 
    """Reads NYC borough boundaries."""
    nyc_boroughs = gpd.read_file(os.path.join(data_path,'Borough Boundaries',
                                          'geo_export_0b51ec0c-deb3-471e-8e50-88fa14797859.shp'))
    return nyc_boroughs


def save_manhattan_rides(man_rides,processed_pkl):
    """Save taxi rides dataframe to pickle file"""
    pickle.dump(man_rides,open(processed_pkl,"wb"))
    return


def load_manhattan_rides(processed_pkl): 
    """Load dataframe from pickle file"""
    mdf = pickle.load(open(processed_pkl,"rb"))
    return mdf


def prepare_ride_data(original_csv): 
    """Reads in all ride data for a single month, returns a dataframe with only rides that start and end
    in Manhattan. 
    """
    
    def create_empty_man_rides():
        """Create empty dataframe, which will soon store info on all rides in Manhattan"""
        return pd.DataFrame({col:[] for col in ['VendorID','tpep_pickup_datetime','tpep_dropoff_datetime',
                            'passenger_count','trip_distance','pickup_point','dropoff_point','geometry']})

    def chunk_to_gdf(taxi_rides): 
        """Converts chunk from initial CSV file into a geodataframe."""
        from external_variables import default_crs
        taxi_rides['pickup_point'] = taxi_rides.apply(lambda row: Point(row['pickup_longitude'],
                                                                        row['pickup_latitude']),axis=1)
        taxi_rides['dropoff_point'] = taxi_rides.apply(lambda row: Point(row['dropoff_longitude'],
                                                                         row['dropoff_latitude']),axis=1)
        taxi_rides['geometry']=taxi_rides['dropoff_point']
        taxi_rides = gpd.GeoDataFrame(taxi_rides)
        taxi_rides.crs = {'init':default_crs}
        taxi_rides = taxi_rides.drop(['pickup_longitude','pickup_latitude',
                                      'dropoff_longitude','dropoff_latitude'],axis=1)
        return taxi_rides
    
    def select_manhattan_rides(chunk_rides):
        """Returns dataframe that only contains rides that start and end in Manhattan"""
        return chunk_rides[(chunk_rides['pickup_boro']=='Manhattan') & 
                           (chunk_rides['dropoff_boro']=='Manhattan')]
    
    def find_pickup_dropoff_boro(taxi_rides,nyc_boroughs): 
        """Determine the pickup and dropoff boroughs using spatial joins."""
        taxi_rides = taxi_rides.set_geometry('pickup_point')
        taxi_rides = gpd.sjoin(taxi_rides,nyc_boroughs[['boro_name','geometry']],op='within')\
                        .drop('index_right',axis=1).rename(mapper={'boro_name':'pickup_boro'},axis=1)
        taxi_rides = taxi_rides.set_geometry('dropoff_point')
        taxi_rides = gpd.sjoin(taxi_rides,nyc_boroughs[['boro_name','geometry']],op='within')\
                        .drop('index_right',axis=1).rename(mapper={'boro_name':'dropoff_boro'},axis=1)
        return taxi_rides
    
    # Prepare to load CSV file in chunks
    csv_cols = ['VendorID','tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','trip_distance',
                'pickup_longitude','pickup_latitude','pickup_boro','dropoff_boro',
                'dropoff_longitude','dropoff_latitude']
    taxi_rides = pd.read_csv(original_csv,parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'],
                             usecols=csv_cols,chunksize=1e5)

    # Save all rides, from each chunk, that start and end in Manhattan to a single dataframe. 
    man_rides = create_empty_man_rides()
    for chunk in taxi_rides: 
        chunk_rides = chunk_to_gdf(chunk)
        chunk_rides = find_pickup_dropoff_boro(chunk_rides,nyc_boroughs)
        man_chunk = select_manhattan_rides(chunk_rides)
        man_rides = man_rides.append(man_chunk,sort=False)
        
    return man_rides


def plot_pickup_dropoff_points(man_rides,nyc_boroughs): 
    """Plot pickup and dropoff locations for a subset of rides"""
    manhattan_shape = nyc_boroughs['geometry'][nyc_boroughs['boro_name']=='Manhattan']
    fig = plt.figure(figsize=[12,8])
    man_rides['geometry'] = man_rides['pickup_point']
    ax = fig.add_subplot(121)
    man_rides.plot(ax=ax,markersize=5,alpha=0.15)
    manhattan_shape.plot(ax=ax,color='k',alpha=0.25)
    ax.set_title('Pickup')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    man_rides['geometry'] = man_rides['dropoff_point']
    ax = fig.add_subplot(122)
    man_rides.plot(ax=ax,markersize=5,alpha=0.15)
    manhattan_shape.plot(ax=ax,color='k',alpha=0.25)
    ax.set_title('Dropoff')
    ax.set_xlabel('Longitude')
    return fig

def create_manhattan_grid(nyc_boroughs,delta_x):
    """Creates geodataframe with grid boxes over Manhattan
    
    ***********
    PARAMETERS: 
    nyc_boroughs = geodataframe with borough boundaries (including Manhattan)
    delta_x      = Approximate width of grid boxes (in miles)
    """
    
    def get_manhattan_bounds(nyc_boroughs):
        """Gets lat/lon boundaries of Manhattan polygon."""
        manhattan_shape = nyc_boroughs['geometry'][nyc_boroughs['boro_name']=='Manhattan']
        return manhattan_shape.bounds
    
    def create_manhattan_grid_boxes(bounds_object,delta_x):
        """Returns a list of grid boxes (length,width~delta_x) that cover Manhattan. 
        NOTE: The 53 and 69 in the second and third lines are for converting miles->lat/lon degrees at ~40N.
        """
        minx,maxx,miny,maxy = manhattan_bounds['minx'].iloc[0],manhattan_bounds['maxx'].iloc[0],\
                              manhattan_bounds['miny'].iloc[0],manhattan_bounds['maxy'].iloc[0]
        grid_x = np.linspace(minx,maxx,int((maxx-minx)//(delta_x/53)))
        grid_y = np.linspace(miny,maxy,int((maxy-miny)//(delta_x/69)))
        result = []
        for i,_ in enumerate(grid_x[:-1]):
            for j,_ in enumerate(grid_y[:-1]):
                b = box(grid_x[i], grid_y[j], grid_x[i+1], grid_y[j+1])
                result.append(b)
        return result
    
    def manhattan_boxes_to_gdf(boxes,manhattan_shape):
        """Creates a geodataframe with the grid box polygons and indices (will be used as unique IDs)"""
        from external_variables import default_crs
        manhattan_grid = gpd.GeoDataFrame(pd.DataFrame({
            'grid_idx': list(range(len(boxes))),
            'geometry': boxes
        }))
        manhattan_grid.crs={'init':default_crs}
        manhattan_grid = gpd.sjoin(manhattan_grid,gpd.GeoDataFrame(manhattan_shape),op='intersects')\
                            .drop('index_right',axis=1)
        return manhattan_grid
    
    manhattan_shape = nyc_boroughs['geometry'][nyc_boroughs['boro_name']=='Manhattan']
    manhattan_bounds = get_manhattan_bounds(nyc_boroughs)
    boxes = create_manhattan_grid_boxes(manhattan_bounds,delta_x)
    manhattan_grid = manhattan_boxes_to_gdf(boxes,manhattan_shape)
    
    return manhattan_grid

def calc_pickup_dropoff_gridbox(man_rides,manhattan_grid): 
    """Identifies grid boxes in which each pickup/dropoff occurred."""
    man_rides['geometry'] = man_rides['pickup_point']
    man_rides = gpd.sjoin(man_rides,manhattan_grid,op='within').drop('index_right',axis=1)
    man_rides = man_rides.rename(mapper={'grid_idx':'pickup_box'},axis=1)
    man_rides['geometry'] = man_rides['dropoff_point']
    man_rides = gpd.sjoin(man_rides,manhattan_grid,op='within').drop('index_right',axis=1)
    man_rides = man_rides.rename(mapper={'grid_idx':'dropoff_box'},axis=1)
    return man_rides

def identify_unnecessary_rides_box1_box2(data,delta_t,n_seats): 
    """Returns a list of rides from box1->box2 that could have been eliminated through ridesharing, 
    according to my algorithm (see the Jupyter notebook for a description of the algorithm)
    
    ***********
    PARAMETERS: 
    data    = dataframe with all rides from box1->box2
    delta_t = maximum allowable waiting time
    n_seats = number of passenger seats in a single ridesharing vehicle. 
    """
    
    # Determine the column number for each relevant column -- necessary for position-based indexing
    col_pickuptime = list(data.columns).index('tpep_pickup_datetime')
    col_pass_count = list(data.columns).index('passenger_count')

    # Pre-sort the data by the pickup time. 
    data = data.sort_values(by='tpep_pickup_datetime')  

    # Pre-compute some stuff
    num_rides = np.shape(data)[0]                        # Number of rides from box1->box2
    five_mins = pd.Timedelta(str(delta_t)+' minutes')    # 5-minute timedelta

    # Initialize empty storage structures
    saved_rides = data[data['tpep_pickup_datetime']==-1] # Empty dataframe to store saved rides
    accounted_for = np.zeros((np.shape(data)[0],1))      # Zero vector: indicates if rides have been accounted for.

    # Algorithm implementation (see docstring for algorithm description)
    for i in range(num_rides): 
        if accounted_for[i]==0: 
            accounted_for[i] = 1
            num_in_car,j = data.iloc[i,col_pass_count],1
            while (i+j<=num_rides-1 and data.iloc[i+j,col_pickuptime]-data.iloc[i,col_pickuptime]<=five_mins): 
                if num_in_car+data.iloc[i+j,col_pass_count]<=n_seats and not accounted_for[i+j]: 
                    saved_rides = pd.concat([saved_rides,data.iloc[i+j:i+j+1,:]])
                    num_in_car+=data.iloc[i+j,col_pass_count]
                    accounted_for[i+j]=1
                j+=1

    return saved_rides

def identify_all_unnecessary_rides(man_rides,delta_t,n_seats): 
    """Identifies saved rides in the entire city, then saves info on those rides to one geodataframe."""
    all_saved_rides = man_rides[man_rides['tpep_pickup_datetime']==-1]
    for idx,data in man_rides.groupby(['pickup_box','dropoff_box']): 
        saved_rides = identify_unnecessary_rides_box1_box2(data,delta_t,n_seats)
        all_saved_rides = pd.concat([all_saved_rides,saved_rides])
    return all_saved_rides

def select_first_week(taxi_rides): 
    """Select only rides that fall within the first week of June (June 6-12, 2016)."""
    taxi_rides = taxi_rides[taxi_rides['tpep_pickup_datetime'].dt.day.isin([6,7,8,9,10,11,12])]
    return taxi_rides

def plot_manhattan_grid(manhattan_grid,nyc_boroughs): 
    """Plots grid over the shape of Manhattan"""
    manhattan_shape = nyc_boroughs['geometry'][nyc_boroughs['boro_name']=='Manhattan']
    fig = plt.figure(figsize=[5,6])
    ax = fig.add_subplot(111)
    manhattan_shape.plot(ax=ax,color='gray',alpha=1)
    manhattan_grid.plot(ax=ax,color='white',edgecolor='k',linewidth=1,alpha=0.5) 
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Manhattan Grid')
    plt.tight_layout()
    return fig

def plot_unnecessary_ride_pickup_locations(all_saved_rides,nyc_boroughs): 
    """Plots the pickup location for each saved ride"""
    manhattan_shape = nyc_boroughs['geometry'][nyc_boroughs['boro_name']=='Manhattan']
    all_saved_rides['geometry'] = all_saved_rides['pickup_point']
    fig = plt.figure(figsize=[9,10.5])
    ax = fig.add_subplot(111)
    manhattan_shape.plot(ax=ax,color='gray',alpha=0.5)
    all_saved_rides.plot(ax=ax,alpha=0.1,markersize=2)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Pickup locations for unnecessary rides')
    return fig

def plot_daily_efficiency(daily_effs): 
    fig = plt.figure(figsize=[8,5])
    ax = fig.add_subplot(111)
    daily_effs.plot(ax=ax,color='black')
    ax.set_xlabel('Date')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Daily efficiency, June 6-12, 2016')
    ax.set_ylim([0,8])
    return fig

def plot_hourly_efficiency(hourly_effs): 
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111)
    hourly_effs.plot(ax=ax,color='black')
    ax.set_xlabel('Hours since 0:00 EST on June 6, 2016')
    ax.set_ylabel('Hourly efficiency (%)')
    ax.set_title('Houly efficiencies for week of June 6-12, 2016')
    ax.set_xticks(range(0,168,24))
    ax.set_ylim([0,12])
    ax.grid(b=True, which='minor', color='k', linestyle='-',alpha=0.2)
    return fig

def plot_hourly_ride_counts(hourly_ride_counts): 
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111)
    hourly_ride_counts.plot(ax=ax,color='black')
    ax.set_xlabel('Hours since 0:00 EST on June 6, 2016')
    ax.set_ylabel('Hourly number of rides')
    ax.set_title('Houly Manhattan-to-Manhattan ride counts for week of June 6-12, 2016')
    ax.set_xticks(range(0,168,24))
    ax.set_ylim([0,22500])
    ax.grid(b=True, which='minor', color='k', linestyle='-',alpha=0.2)
    return fig

def efficiency_gridbox_choropleth(gridded_effs,nyc_boroughs): 
    manhattan_shape = nyc_boroughs['geometry'][nyc_boroughs['boro_name']=='Manhattan']
    fig = plt.figure(figsize=[9,10.5])
    ax = fig.add_subplot(111)
    manhattan_shape.plot(ax=ax,color='gray',alpha=0.5)
    gridded_effs['geometry'] = gridded_effs['geometry'].intersection(manhattan_shape.unary_union)
    gridded_effs.plot(ax=ax,alpha=0.65,column='efficiency',legend=True,cmap='viridis',
                                                                      edgecolor='k',linewidth=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Efficiency, week of June 6-12')
    return fig
