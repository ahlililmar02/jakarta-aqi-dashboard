import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

# Define your SQL query to get only relevant data
query = """
SELECT *
FROM aqi
ORDER BY time asc
"""

# Read into DataFrame
df_pm25_daily = pd.read_sql_query(query, conn)

# Close connection
conn.close()


# In[4]:


def normalize_coords(df, lon_col='longitude', lat_col='latitude'):
    df[lon_col] = df[lon_col].round(4)   # ~11m precision
    df[lat_col] = df[lat_col].round(4)
    return df

df_pm25_daily = normalize_coords(df_pm25_daily)


# In[5]:


import pandas as pd

# Ensure 'time' is datetime
df_pm25_daily['time'] = pd.to_datetime(df_pm25_daily['time'])

# Create a 'date' column
df_pm25_daily['date'] = df_pm25_daily['time'].dt.date

# Aggregate by date (e.g., mean of pm25)
df_pm25_daily = df_pm25_daily.groupby(['station', 'date']).agg({
    'aqi': 'mean',
    'pm25': 'mean',
    'latitude': 'first',
    'longitude': 'first',
    'sourceid': 'first'
}).reset_index()

# Filter for 2025
df_pm25_daily = df_pm25_daily[
    (pd.to_datetime(df_pm25_daily['date']).dt.year == 2025)
]

df_pm25_daily.to_csv("daily_pm25.csv", index=False)


# In[6]:


import ee
from google.auth import credentials
import pandas as pd
import matplotlib.pyplot as plt

# Set your actual project ID here (from Google Cloud Console)
project_id = 'ee-dashboardaq'  # 🔁 Change this

# Service account details
service_account = 'dashboard-aq@ee-dashboardaq.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'earthengineapikey.json')

# ✅ Initialize with project ID
ee.Initialize(credentials, project=project_id)

print("✅ Earth Engine initialized with project:", project_id)


# In[7]:


from datetime import datetime, timedelta

# Convert 'time' to datetime

# Set start and end dates
start_date = datetime.strptime("2025-02-03", "%Y-%m-%d").date()
end_date = df_pm25_daily['date'].max()

print(f"🗓️ Processing from {start_date} to {end_date}")

date_range = pd.date_range(start=start_date, end=end_date)


# In[8]:


roi = ee.Geometry.Rectangle([106.56, -6.5, 107.05, -6.0])


# In[9]:


import ee
import pandas as pd

# Create full date range

# Result container
all_results = []

# Loop through each date
for date in date_range:
    print(f"📆 Processing date: {date.date()}")
    try:
        next_day = date + timedelta(days=1)

        #AOD
        aod_img = (
            ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
            .filterDate(str(date.date()), str(next_day.date()))
            .filterBounds(roi)
            .select(['Optical_Depth_047', 'AOD_QA'])
            .map(lambda img: img.updateMask(img.select('AOD_QA').gte(1)))
            .select('Optical_Depth_047')
            .mean()
            .multiply(0.001)
            .rename("mean_AOD")
        )

        # CO
        co_img = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO")
            .filterDate(str(date.date()), str(next_day.date()))
            .filterBounds(roi)
            .select("CO_column_number_density")
            .mean()
            .rename("mean_CO")
        )

        # NO2
        no2_img = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
            .filterDate(str(date.date()), str(next_day.date()))
            .filterBounds(roi)
            .select("NO2_column_number_density")
            .mean()
            .rename("mean_NO2")
        )

        so2_img = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
            .filterDate(str(date.date()), str(next_day.date()))
            .filterBounds(roi)
            .select("SO2_column_number_density")
            .mean()
            .rename("mean_SO2")
        )

        hcho_img = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_HCHO")
            .filterDate(str(date.date()), str(next_day.date()))
            .filterBounds(roi)
            .select("tropospheric_HCHO_column_number_density")
            .mean()
            .rename("mean_HCHO")
        )

        # Combine all images
        combined = aod_img.addBands([co_img,no2_img,so2_img,hcho_img])

        # Sample across ROI
        sampled = combined.sample(
            region=roi,
            scale=1000,
            geometries=True,
        )

        # Parse features
        features = sampled.getInfo()['features']
        count = 0
        for f in features:
            coords = f['geometry']['coordinates']
            props = f['properties']
            if all(k in props for k in ['mean_AOD','mean_CO', 'mean_NO2','mean_SO2','mean_HCHO']):
                all_results.append({
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'mean_AOD': props['mean_AOD'],
                    'mean_CO': props['mean_CO'],
                    'mean_NO2': props['mean_NO2'],
                    'mean_SO2': props['mean_SO2'],
                    'mean_HCHO': props['mean_HCHO'],
                    'date': date.date()
                })
                count += 1

        print(f"✅ Finished {date.date()} with {count} valid pixels")

    except Exception as e:
        print(f"⚠️ Skipped {date.date()}: {e}")

# Final DataFrame
df_all_in_one = pd.DataFrame(all_results)
print(f"📊 Final dataset shape: {df_all_in_one.shape}")


# In[10]:


df_all = df_all_in_one.groupby('date').filter(lambda x: len(x) > 300)


# In[11]:


elevation_img = ee.Image("USGS/SRTMGL1_003").rename('elevation')

# Sample elevation
elevation_sampled = elevation_img.sampleRegions(
    collection=roi,
    scale=1000,  # adjust scale as needed
    geometries=True
)

# Convert to dataframe
elevation_features = elevation_sampled.getInfo()['features']
elevation_results = []
for f in elevation_features:
    coords = f['geometry']['coordinates']
    props = f['properties']
    if 'elevation' in props:
        elevation_results.append({
            'longitude': coords[0],
            'latitude': coords[1],
            'elevation': props['elevation']
        })

elevation_df = pd.DataFrame(elevation_results)
print("✅ Elevation static shape:", elevation_df.shape)


# In[12]:


import pandas as pd

# Create full date range
date_range_all = pd.date_range(
    start='2025-02-01',
    end= df_pm25_daily['date'].max(),
    freq='D'  # daily frequency
)

print(date_range_all)
print(f"Total days: {len(date_range_all)}")
print(date_range_all.dtype)  # should show datetime64[ns]


# In[13]:


# Extract unique year-month pairs from your date_range
unique_months = sorted(set((d.year, d.month) for d in date_range_all))

# Rebuild a monthly date_range using the first available date in that month
date_range_monthly = [min([d for d in date_range_all if d.year == y and d.month == m]) 
                      for (y, m) in unique_months]


# In[14]:


import ee
import pandas as pd
from datetime import timedelta

# Define MODIS 16-day NDVI collection
modis_ndvi = ee.ImageCollection('MODIS/061/MOD13Q1').select('NDVI')

all_ndvi_results = []

# Loop in 16-day steps
step = 16
for i in range(0, len(date_range), step):
    start_date = date_range[i]
    end_date = start_date + timedelta(days=step)

    print(f"📆 Processing {start_date.date()} to {end_date.date()}")

    try:
        # Filter for 16-day window
        ndvi_img = (
            modis_ndvi
            .filterDate(str(start_date.date()), str(end_date.date()))
            .filterBounds(roi)
            .mean()
            .rename("NDVI")
        )

        # Sample NDVI at ROI
        sampled = ndvi_img.sampleRegions(
            collection=roi,  # your points or ROI collection
            scale=1000,
            geometries=True
        )

        features = sampled.getInfo()['features']
        count = 0
        for f in features:
            coords = f['geometry']['coordinates']
            props = f['properties']
            if 'NDVI' in props:
                all_ndvi_results.append({
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'NDVI': props['NDVI'],
                    'date': start_date.date()
                })
                count += 1

        print(f"✅ Finished {start_date.date()} to {end_date.date()} with {count} NDVI points")

    except Exception as e:
        print(f"⚠️ Skipped {start_date.date()} to {end_date.date()}: {e}")

# Convert to DataFrame
ndvi_df = pd.DataFrame(all_ndvi_results)
ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
print(f"📊 NDVI DataFrame shape: {ndvi_df.shape}")


# In[15]:


# VIIRS monthly collection
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad")

all_viirs_results = []

for start_date in date_range_monthly:
    end_date = (start_date + pd.offsets.MonthEnd(1)).date()

    print(f"📆 Processing {start_date.date()} to {end_date}")

    try:
        # Monthly VIIRS (should return exactly one image per month)
        viirs_img = (
            viirs.filterDate(str(start_date.date()), str(end_date))
                 .filterBounds(roi)
                 .mean()
                 .rename("avg_rad")
        )

        # Sample VIIRS at ROI
        sampled = viirs_img.sampleRegions(
            collection=roi,
            scale=1000,
            geometries=True
        )

        features = sampled.getInfo()["features"]
        count = 0
        for f in features:
            coords = f["geometry"]["coordinates"]
            props = f["properties"]
            if "avg_rad" in props:
                all_viirs_results.append({
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "avg_rad": props["avg_rad"],
                    "date": start_date.date()
                })
                count += 1

        print(f"✅ Finished {start_date.date()} with {count} VIIRS points")

    except Exception as e:
        print(f"⚠️ Skipped {start_date.date()}: {e}")

# Convert to DataFrame
viirs_df = pd.DataFrame(all_viirs_results)
print(f"📊 VIIRS DataFrame shape: {viirs_df.shape}")


# In[16]:


# Built-up dataset
ghsl = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
img = ghsl.filter(ee.Filter.eq('system:index', '2025')).first()
built = img.select('built_surface').rename('built_surface')

# Sample
built_sampled = built.sampleRegions(
    collection=roi,
    scale=1000,
    geometries=True
)

# Convert to DataFrame
features = built_sampled.getInfo()['features']
all_built = []
for f in features:
    coords = f['geometry']['coordinates']
    props = f['properties']
    all_built.append({
        'longitude': coords[0],
        'latitude': coords[1],
        'built_surface_m2': props['built_surface'],
        'year': 2025
    })

df_built = pd.DataFrame(all_built)
print("Final built-up DataFrame shape:", df_built.shape)


# In[17]:


import pandas as pd
import ee

# Sentinel-2 SR Collection
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 55))  # filter cloud cover
)

all_swir_nir_results = []

# Define seasonal ranges (3-month frequency)
#date_range_seasonal = pd.date_range("2025-01-01", "2025-12-31", freq="3MS")  # start of each season

for start_date in date_range_monthly:
    end_date = (start_date + pd.offsets.MonthEnd(1)).date()

    print(f"📆 Processing {start_date.date()} to {end_date}")

    try:
        # Seasonal composite: median over 3 months
        s2_img = (
            s2.filterDate(str(start_date.date()), str(end_date))
              .median()
              .select(["B8", "B11"])
        )

        # Sample at ROI
        sampled = s2_img.sampleRegions(
            collection=roi,
            scale=1000,  # Sentinel-2 resolution
            geometries=True
        )

        features = sampled.getInfo()["features"]
        count = 0
        for f in features:
            coords = f["geometry"]["coordinates"]
            props = f["properties"]
            if "B8" in props and "B11" in props:
                all_swir_nir_results.append({
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "nir": props["B8"],
                    "swir": props["B11"],
                    "date": start_date.date()
                })
                count += 1

        print(f"✅ Finished {start_date.date()} with {count} SWIR/NIR points")

    except Exception as e:
        print(f"⚠️ Skipped {start_date.date()}: {e}")

# Convert to DataFrame
swir_nir_df = pd.DataFrame(all_swir_nir_results)

# Calculate NDBI in pandas
swir_nir_df["ndbi"] = (
    (swir_nir_df["swir"] - swir_nir_df["nir"]) /
    (swir_nir_df["swir"] + swir_nir_df["nir"])
)

# Add season label (DJF, MAM, JJA, SON)
def get_season(date):
    m = date.month
    if m in [12, 1, 2]:
        return "DJF"
    elif m in [3, 4, 5]:
        return "MAM"
    elif m in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"

swir_nir_df["season"] = swir_nir_df["date"].apply(get_season)

print(f"📊 SWIR/NIR DataFrame shape: {swir_nir_df.shape}")



# In[18]:


import ee
import pandas as pd


modis_lst = ee.ImageCollection('MODIS/061/MOD21C3').select(['LST_Day'])

all_lst_results = []

# Loop by month
for single_date in date_range_monthly:
    start_date = single_date
    end_date = (single_date + pd.offsets.MonthEnd(1)).date()

    print(f"📆 Processing {start_date.date()} to {end_date}")

    try:
        # Filter for the month
        lst_img = (
            modis_lst
            .filterDate(str(start_date.date()), str(end_date))
            .filterBounds(roi)
            .mean()
            .subtract(273.15)  # Kelvin → Celsius
            .rename(['LST_Day'])
        )
        
        lst_img = lst_img.updateMask(lst_img.gt(-50).And(lst_img.lt(60)))


        # Sample LST at ROI
        sampled = lst_img.sampleRegions(
            collection=roi,
            scale=1000,
            geometries=True
        )

        features = sampled.getInfo()['features']
        count = 0
        for f in features:
            coords = f['geometry']['coordinates']
            props = f['properties']
            if 'LST_Day' in props:
                all_lst_results.append({
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'lst': props['LST_Day'],
                    'month': start_date.strftime('%Y-%m')
                })
                count += 1

        print(f"✅ Finished {start_date.date()} to {end_date} with {count} LST points")

    except Exception as e:
        print(f"⚠️ Skipped {start_date.date()} to {end_date}: {e}")

# Convert to DataFrame
lst_monthly_df = pd.DataFrame(all_lst_results)
print(f"📊 Monthly LST DataFrame shape: {lst_monthly_df.shape}")


# In[19]:


import ee
import pandas as pd


modis_lst = ee.ImageCollection('MODIS/061/MOD21A1D').select(['LST_1KM'])

all_lst_results = []

# Loop by month
for single_date in date_range_monthly:
    start_date = single_date
    end_date = (single_date + pd.offsets.MonthEnd(1)).date()

    print(f"📆 Processing {start_date.date()} to {end_date}")

    try:
        # Filter for the month
        lst_img = (
            modis_lst
            .filterDate(str(start_date.date()), str(end_date))
            .filterBounds(roi)
            .mean()
            .subtract(273.15)  # Kelvin → Celsius
            .rename(['LST_1KM'])
        )

        lst_img = lst_img.updateMask(lst_img.gt(-50).And(lst_img.lt(60)))

        # Sample LST at ROI
        sampled = lst_img.sampleRegions(
            collection=roi,
            scale=1000,
            geometries=True
        )

        features = sampled.getInfo()['features']
        count = 0
        for f in features:
            coords = f['geometry']['coordinates']
            props = f['properties']
            if 'LST_1KM' in props:
                all_lst_results.append({
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'lst': props['LST_1KM'],
                    'month': start_date.strftime('%Y-%m')
                })
                count += 1

        print(f"✅ Finished {start_date.date()} to {end_date} with {count} LST points")

    except Exception as e:
        print(f"⚠️ Skipped {start_date.date()} to {end_date}: {e}")

# Convert to DataFrame
lst_daily_df = pd.DataFrame(all_lst_results)
print(f"📊 Monthly LST DataFrame shape: {lst_daily_df.shape}")


# In[20]:


# Concatenate both dataframes
lst_df = pd.concat([lst_daily_df, lst_monthly_df], ignore_index=True)

# Drop duplicates on lat, lon, month, keeping the first occurrence (daily preferred)
lst_df = lst_df.drop_duplicates(
    subset=["longitude", "latitude", "month"],
    keep="first"  # keep daily if duplicate exists
).reset_index(drop=True)

# Drop February 2025
lst_df = lst_df[lst_df["month"] != "2025-02"].reset_index(drop=True)
lst_df = lst_df[lst_df["month"] != "2025-09"].reset_index(drop=True)


# In[21]:


import pandas as pd

# Example get_season function
def get_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"

# Convert 'month' to datetime and extract season
lst_df["season"] = pd.to_datetime(lst_df["month"]).dt.month.apply(get_season)

# Aggregate LST per latitude, longitude, and season
seasonal_lst = (
    lst_df.groupby(["latitude", "longitude", "season"])["lst"]
    .mean()
    .reset_index()
)


# In[22]:


# WorldPop 2020 for Indonesia (change country code as needed)
worldpop = ee.Image("WorldPop/GP/100m/pop/IDN_2020").rename("population")

# Sample over ROI
worldpop_sampled = worldpop.sampleRegions(
    collection=roi,     # your FeatureCollection or Geometry
    scale=1000,          # WorldPop native resolution is ~100m
    geometries=True
)

# Convert to DataFrame
features = worldpop_sampled.getInfo()['features']
all_pop = []
for f in features:
    coords = f['geometry']['coordinates']
    props = f['properties']
    all_pop.append({
        'longitude': coords[0],
        'latitude': coords[1],
        'population': props['population'],
        'year': 2020
    })

df_worldpop = pd.DataFrame(all_pop)
print("Final WorldPop DataFrame shape:", df_worldpop.shape)


# In[ ]:


import osmnx as ox
import geopandas as gpd
import shapely.geometry as geom
import numpy as np
import pandas as pd

# -------------------------------
# 1. Define ROI (Jakarta bounding box)
# -------------------------------
min_lon, min_lat, max_lon, max_lat = 106.56, -6.5, 107.05, -6.0
roi_polygon = geom.box(min_lon, min_lat, max_lon, max_lat)

# -------------------------------
# 2. Download OSM roads
# -------------------------------
print("📥 Downloading OSM roads...")
roads = ox.features_from_polygon(
    roi_polygon,
    tags={'highway': True}  # filter for road types
)

# Keep only LineString geometries
roads = roads[roads.geometry.type.isin(['LineString', 'MultiLineString'])]
print(f"✅ Roads extracted: {len(roads)}")

# -------------------------------
# 3. Create grid (~1 km = 0.01 deg near Jakarta)
# -------------------------------
step = 0.01
lons = np.arange(min_lon, max_lon, step)
lats = np.arange(min_lat, max_lat, step)

grid_cells = []
for x in lons:
    for y in lats:
        grid_cells.append(geom.box(x, y, x+step, y+step))

grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
print(f"✅ Grid cells created: {len(grid)}")

# -------------------------------
# 4. Spatial join: assign roads to grid cells
# -------------------------------
roads = roads.to_crs("EPSG:4326")
grid = grid.to_crs("EPSG:4326")

# Explode MultiLineStrings into LineStrings for length calculation
roads = roads.explode(index_parts=True).reset_index(drop=True)

# Spatial join: which grid each road belongs to
joined = gpd.sjoin(roads[['geometry']], grid.reset_index(), how="inner", predicate="intersects")

# -------------------------------
# 5. Calculate road length per grid
# -------------------------------
joined = joined.to_crs("EPSG:3857")  # project to meters
joined['length_m'] = joined.length

# Aggregate road length per grid cell
road_length = joined.groupby("index")['length_m'].sum()

# Add results back to grid
grid['road_length_m'] = grid.index.map(road_length).fillna(0)

# -------------------------------
# 6. Compute road density (km/km²)
# -------------------------------
grid_m = grid.to_crs("EPSG:3857")  # project grid to meters
grid['cell_area_m2'] = grid_m.area

grid['road_density'] = (
    (grid['road_length_m'] / 1000) / (grid['cell_area_m2'] / 1e6)
).fillna(0)

# -------------------------------
# 7. Convert to tabular format
# -------------------------------
grid['longitude'] = grid.centroid.x
grid['latitude'] = grid.centroid.y

df_osm = grid[['longitude', 'latitude', 'road_density']].copy()
print("✅ Final OSM Road Density DataFrame shape:", df_osm.shape)


# In[25]:


ndvi_df["NDVI"]= ndvi_df["NDVI"]/10000


# In[26]:


def expand_to_daily(df_16day, start_date, end_date, cols):
    # Ensure date column is datetime
    df_16day['date'] = pd.to_datetime(df_16day['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    expanded = []
    for (lon, lat), group in df_16day.groupby(['longitude', 'latitude']):
        # Full daily date range
        full_range = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')})
        # Merge with group
        g = pd.merge(full_range, group[['date'] + cols], on='date', how='left')
        # Fill missing values
        g[cols] = g[cols].ffill().bfill()
        g['longitude'] = lon
        g['latitude'] = lat
        expanded.append(g)
        
    return pd.concat(expanded, ignore_index=True)

# Expand NDVI and NDBI to daily
ndvi_daily = expand_to_daily(ndvi_df, df_all['date'].min(), df_all['date'].max(), ['NDVI'])
ndbi_daily = expand_to_daily(swir_nir_df, df_all['date'].min(), df_all['date'].max(), ['ndbi'])


# In[27]:


def expand_seasonal_to_daily(df_seasonal, start_date, end_date, col):
    expanded = []
    for (lon, lat), group in df_seasonal.groupby(['longitude', 'latitude']):
        # Full daily range
        full_range = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')})
        full_range["season"] = full_range["date"].dt.month.apply(get_season)

        # Merge with seasonal values
        g = pd.merge(full_range, group[['season', col]], on='season', how='left')

        # Add coordinates
        g['longitude'] = lon
        g['latitude'] = lat
        expanded.append(g)

    return pd.concat(expanded, ignore_index=True)
seasonal_daily_lst = expand_seasonal_to_daily(seasonal_lst, df_all["date"].min(), df_all["date"].max(), col="lst")
seasonal_daily_lst = seasonal_daily_lst.dropna()


# In[28]:


def merge_16day(base_df, df_16day, col_names):
    # Ensure datetime
    base_df['date'] = pd.to_datetime(base_df['date'])
    df_16day['date'] = pd.to_datetime(df_16day['date'])

    # Merge
    df = pd.merge(
        base_df,
        df_16day[['longitude','latitude','date']+col_names],
        on=['longitude','latitude','date'],
        how='left'
    )

    # Sort by lon, lat, date for fill
    df = df.sort_values(['longitude','latitude','date'])

    
    return df   # ✅ make sure to return

from scipy.spatial import cKDTree

def merge_nearest(base_df, ref_df, value_col, max_dist=0.02):
    # KDTree for fast nearest lookup
    tree = cKDTree(ref_df[['longitude','latitude']].values)
    dist, idx = tree.query(base_df[['longitude','latitude']].values, k=1)

    nearest_vals = [ref_df.iloc[i][value_col] if d <= max_dist else None 
                    for i, d in zip(idx, dist)]
    base_df[value_col] = nearest_vals
    return base_df

def merge_static(base_df, df_static, col_names):
    # Merge on lon/lat only
    df = pd.merge(
        base_df,
        df_static[['longitude','latitude']+col_names],
        on=['longitude','latitude'],
        how='left'
    )
    return df


# In[29]:


df_sat = merge_16day(df_all, ndvi_daily, ['NDVI'])
df_sat = merge_16day(df_sat, ndbi_daily, ['ndbi'])
df_sat = merge_16day(df_sat, seasonal_daily_lst, ['lst'])
df_sat = merge_nearest(df_sat, elevation_df, 'elevation', max_dist=0.02) 
df_sat = merge_nearest(df_sat, df_worldpop, 'population', max_dist=0.02) 
df_sat = merge_nearest(df_sat, df_built, 'built_surface_m2', max_dist=0.02) 
df_sat = merge_nearest(df_sat, df_osm, 'road_density', max_dist=0.02) 
df_sat = merge_static(df_sat, viirs_df, ['avg_rad'])
df_sat = df_sat.dropna()


# In[ ]:


all_results = []

# Get unique dates from df_sat
df_sat['date'] = pd.to_datetime(df_sat['date'])

# Get sorted unique dates
unique_dates = pd.Series(df_sat["date"].unique()).sort_values().to_list()

# Loop through each unique date
for date in date_range:
    print(f"📆 Processing date: {date.date()}")
    try:
        next_day = date + timedelta(days=1)

        # ERA5-Land Daily
        era5_img = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(str(date.date()), str(next_day.date()))
            .filterBounds(roi)
            .select([
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
                "temperature_2m",
                "dewpoint_temperature_2m",
                "surface_pressure",
                "total_precipitation_sum"
            ])
            .mean()
        )

        era5_img_resampled = era5_img \
            .resample('bilinear') \
            .reproject(crs='EPSG:4326', scale=1000)

        # Sample ERA5 values at pollutant points
        sampled = era5_img_resampled.sampleRegions(
            collection=roi,
            scale=1000,
            geometries=True
        )
        # Parse features
        features = sampled.getInfo()['features']
        count = 0
        for f in features:
            coords = f['geometry']['coordinates']
            props = f['properties']
            all_results.append({
                'longitude': coords[0],
                'latitude': coords[1],
                'u10': props.get('u_component_of_wind_10m'),
                'v10': props.get('v_component_of_wind_10m'),
                't2m': props.get('temperature_2m', 0) - 273.15 if props.get('temperature_2m') is not None else None,  # Kelvin → °C
                'd2m': props.get('dewpoint_temperature_2m', 0) - 273.15 if props.get('dewpoint_temperature_2m') is not None else None,
                'sp': props.get('surface_pressure'),
                'tp': props.get('total_precipitation_sum'),
                'date': date.date()
            })
            count += 1

        print(f"✅ Finished {date.date()} with {count} ERA5-Land pixels")

    except Exception as e:
        print(f"⚠️ Skipped {date.date()}: {e}")

# Final DataFrame
df_era5_land = pd.DataFrame(all_results)
print(f"📊 Final dataset shape: {df_era5_land.shape}")

import numpy as np
df_era5 = df_era5_land
df_era5["ws"] = np.sqrt(df_era5["u10"]**2 + df_era5["v10"]**2)


def calculate_relative_humidity(temp_c, dewpoint_c):
    """Returns RH in %"""
    a, b = 17.625, 243.04
    alpha = (a * dewpoint_c) / (b + dewpoint_c)
    beta = (a * temp_c) / (b + temp_c)
    rh = 100 * np.exp(alpha - beta)
    return np.clip(rh, 0, 100)  # Ensure RH is between 0 and 100

df_era5['rh'] = calculate_relative_humidity(df_era5['t2m'], df_era5['d2m'])
df_era = df_era5.drop(columns=['u10','v10','d2m'])


# In[32]:


def merge_date(base_df, df_merge, col_names):
    # Merge on lon/lat only
    df = pd.merge(
        base_df,
        df_merge[['longitude','latitude','date']+col_names],
        on=['longitude','latitude','date'],
        how='left'
    )
    return df

df_era['date'] = pd.to_datetime(df_era['date'])
df_sat['date'] = pd.to_datetime(df_sat['date'])
df_sat = merge_date(df_sat, df_era, ['t2m','rh','ws','tp','sp'])

df_era = merge_16day(df_era, ndvi_daily, ['NDVI'])
df_era = merge_16day(df_era, ndbi_daily, ['ndbi'])
df_era = merge_16day(df_era, seasonal_daily_lst, ['lst'])
df_era = merge_nearest(df_era, elevation_df, 'elevation', max_dist=0.02) 
df_era = merge_nearest(df_era, df_worldpop, 'population', max_dist=0.02) 
df_era = merge_nearest(df_era, df_built, 'built_surface_m2', max_dist=0.02) 
df_era = merge_nearest(df_era, df_osm, 'road_density', max_dist=0.02) 
df_era = merge_static(df_era, viirs_df, ['avg_rad'])
df_era = df_era.dropna()

# In[33]:


df_sat.to_csv("df_sat_final.csv",index=False)
df_era.to_csv("df_era_final.csv",index=False)
