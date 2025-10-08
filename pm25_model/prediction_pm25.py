import os
import pandas as pd
from scipy.spatial import cKDTree


df_pm25_daily = pd.read_csv("daily_pm25.csv")
df_sat = pd.read_csv("df_sat_final.csv")

# First, make sure the date columns are datetime
df_pm25_daily["date"] = pd.to_datetime(df_pm25_daily["date"])
df_sat["date"] = pd.to_datetime(df_sat["date"])

# Initialize a list to store results
results = []

# Loop through each unique date in PM2.5 s
for dt in df_pm25_daily["date"].unique():
    # Filter PM2.5 and satellite data for this date
    pm25_day = df_pm25_daily[df_pm25_daily["date"] == dt].copy()
    sat_day = df_sat[df_sat["date"] == dt].copy()
    
    if sat_day.empty:
        continue  # skip if no satellite data for this date
    
    # Build KDTree for satellite coordinates of this day
    tree = cKDTree(sat_day[["longitude", "latitude"]].values)
    
    # Query nearest satellite point for each PM2.5 station
    pm25_coords = pm25_day[["longitude", "latitude"]].values
    distances, indices = tree.query(pm25_coords, k=1)
    
    # Extract nearest satellite points
    nearest_sat = sat_day.iloc[indices].reset_index(drop=True)
    
    # Merge results
    pm25_day["nearest_sat_lon"] = nearest_sat["longitude"].values
    pm25_day["nearest_sat_lat"] = nearest_sat["latitude"].values
    pm25_day["distance_deg"] = distances
    
    results.append(pm25_day)

# Combine all dates back into a single DataFrame
df_pm25_with_nearest_filtered = pd.concat(results, ignore_index=True)
df_pm25_with_nearest_filtered

# Merge df_pm25_with_nearest_filtered with satellite data
df_merged = df_pm25_with_nearest_filtered.merge(
    df_sat,
    left_on=["date", "nearest_sat_lat", "nearest_sat_lon"],
    right_on=["date", "latitude", "longitude"],
    how="left",
    suffixes=("", "_sat")
)

# Optional: drop redundant latitude/longitude columns from satellite if needed

df_merged = df_merged.drop(columns=['nearest_sat_lon', 'nearest_sat_lat', 'distance_deg', 'longitude_sat','aqi',
       'latitude_sat'])


# In[46]:


# Rebuild X and y from df_merged
drop_cols = [
    'pm25', 'station', 'time', 'date','sourceid',
    'aqi','sample_weight','latitude','longitude','expected_high_pm25','expected_low_pm25'
]
df_merged['date'] = pd.to_datetime(df_merged['date']) 
df_merged['month'] = df_merged['date'].dt.month
df_merged['dayofyear'] = df_merged['date'].dt.dayofyear

df_merged['season'] = df_merged['month'].apply(get_season)

df_merged = df_merged.dropna()
X = df_merged.drop(columns=drop_cols, errors='ignore')
y = df_merged['pm25']

# Encode categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[47]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import xgboost as xgb

# Parameter grid for Randomized Search
param_grid = {
    'n_estimators': [150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 6, 8, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Initialize XGBRegressor
xgb_model = xgb.XGBRegressor(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,            # number of random combinations to try
    scoring='neg_mean_squared_error',  # you can change to 'r2' if you prefer
    cv=5,                 # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search
random_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", random_search.best_params_)
print("Best CV score: ", -random_search.best_score_)
2
# Evaluate on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print("Test R2:", r2_score(y_test, y_pred))


# In[48]:


from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Parameter grid for LightGBM
param_grid_lgbm = {
    'n_estimators': [50, 150, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 6, 8, 10, -1],  # -1 means no limit
    'num_leaves': [20, 31, 50, 70, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.5, 1, 1.5, 2],
    'min_child_samples': [5, 10, 20, 50]
}

# Initialize LGBMRegressor
lgbm_model = lgb.LGBMRegressor(random_state=42)

# Randomized Search
random_search_lgbm = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_grid_lgbm,
    n_iter=50,                  # number of random combinations to try
    scoring='neg_mean_squared_error',  # or 'r2'
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search
random_search_lgbm.fit(X_train, y_train)

# Best parameters
print("Best parameters found for LightGBM: ", random_search_lgbm.best_params_)
print("Best CV score (MSE): ", -random_search_lgbm.best_score_)

# Evaluate on test set
best_model_lgbm = random_search_lgbm.best_estimator_
y_pred_lgbm = best_model_lgbm.predict(X_test)

print("Test R2 (LightGBM):", r2_score(y_test, y_pred_lgbm))
print("Test MSE (LightGBM):", mean_squared_error(y_test, y_pred_lgbm))


# In[49]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Best hyperparameters for Random Forest
best_params_rf = {
    'n_estimators': 150,
    'min_samples_split': 5,
    'min_samples_leaf': 5,
    'max_features': 'log2',
    'max_depth': None,
    'bootstrap': False,
    'random_state': 42
}

# Initialize and train model
best_model_rf = RandomForestRegressor(**best_params_rf)
best_model_rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = best_model_rf.predict(X_test)

# Evaluate
print("Random Forest Test R2:", r2_score(y_test, y_pred_rf))


# In[ ]:


# Start from df_sat
df_estimate = df_sat.copy()
df_estimate['date'] = pd.to_datetime(df_estimate['date'])

# Start from df_sat
# Define bounding box
lon_min, lon_max = 106.5, 107.2
lat_min, lat_max = -6.6, -5.6

# Slice df_merged
df_estimate = df_estimate[
    (df_estimate['longitude'] >= lon_min) &
    (df_estimate['longitude'] <= lon_max) &
    (df_estimate['latitude'] >= lat_min) &
    (df_estimate['latitude'] <= lat_max)
].copy()

# Feature engineering
df_estimate['month'] = df_estimate['date'].dt.month
df_estimate['dayofyear'] = df_estimate['date'].dt.dayofyear
df_estimate['season'] = df_estimate['month'].apply(get_season)
df_estimate = df_estimate.dropna()
# Convert landcover to categorical (int)

# Drop non-feature columns
drop_cols = ['date', 'date_day', 'station_id','sourceid']
X_features = df_estimate.drop(columns=drop_cols, errors='ignore')

# Encode categorical columns
for col in X_features.select_dtypes(include='object').columns:
    X_features[col] = X_features[col].astype('category').cat.codes

# ---- XGB ----
X_xgb = X_features[best_model.get_booster().feature_names]
df_estimate['pm25_xgb'] = best_model.predict(X_xgb)

# ---- LGBM ----
X_lgbm = X_features[best_model_lgbm.feature_name_]
df_estimate['pm25_lgbm'] = best_model_lgbm.predict(X_lgbm)

# ---- RF ----
X_rf = X_features[best_model_rf.feature_names_in_]
df_estimate['pm25_rf'] = best_model_rf.predict(X_rf)


# First, make sure the date columns are datetime
df_pm25_daily["date"] = pd.to_datetime(df_pm25_daily["date"])
df_era["date"] = pd.to_datetime(df_era["date"])

# Initialize a list to store results
results = []

# Loop through each unique date in PM2.5 data
for dt in df_pm25_daily["date"].unique():
    # Filter PM2.5 and satellite data for this date
    pm25_day = df_pm25_daily[df_pm25_daily["date"] == dt].copy()
    sat_day = df_era[df_era["date"] == dt].copy()
    
    if sat_day.empty:
        continue  # skip if no satellite data for this date
    
    # Build KDTree for satellite coordinates of this day
    tree = cKDTree(sat_day[["longitude", "latitude"]].values)
    
    # Query nearest satellite point for each PM2.5 station
    pm25_coords = pm25_day[["longitude", "latitude"]].values
    distances, indices = tree.query(pm25_coords, k=1)
    
    # Extract nearest satellite points
    nearest_sat = sat_day.iloc[indices].reset_index(drop=True)
    
    # Merge results
    pm25_day["nearest_sat_lon"] = nearest_sat["longitude"].values
    pm25_day["nearest_sat_lat"] = nearest_sat["latitude"].values
    pm25_day["distance_deg"] = distances
    
    results.append(pm25_day)

# Combine all dates back into a single DataFrame
df_pm25_with_nearest_filtered = pd.concat(results, ignore_index=True)


# In[57]:


# Merge df_pm25_with_nearest_filtered with satellite data
df_merge_era = df_pm25_with_nearest_filtered.merge(
    df_era,
    left_on=["date", "nearest_sat_lat", "nearest_sat_lon"],
    right_on=["date", "latitude", "longitude"],
    how="left",
    suffixes=("", "_sat")
)

# Optional: drop redundant latitude/longitude columns from satellite if needed

df_merge_era = df_merge_era.drop(columns=['nearest_sat_lon', 'nearest_sat_lat', 'distance_deg', 'longitude_sat',
       'latitude_sat','sourceid'])


# In[64]:


drop_cols = [
    'pm25', 'station', 'time', 'date',
    'aqi','latitude','longitude','expected_high_pm25'
]
df_merge_era['date'] = pd.to_datetime(df_merge_era['date']) 
df_merge_era['month'] = df_merge_era['date'].dt.month
df_merge_era['dayofyear'] = df_merge_era['date'].dt.dayofyear

# Add season feature
def get_season(month):
    if month in [12, 1, 2]:
        return "DJF"   # December–January–February (Rainy season in IDN)
    elif month in [3, 4, 5]:
        return "MAM"   # March–April–May
    elif month in [6, 7, 8]:
        return "JJA"   # June–July–August (Dry season in IDN)
    else:
        return "SON"   # September–October–November

df_merge_era['season'] = df_merge_era['month'].apply(get_season)
df_merge_era = df_merge_era.dropna()
# Rebuild X and y
X = df_merge_era.drop(columns=drop_cols, errors='ignore')
y = df_merge_era['pm25']

# Encode categorical columns
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype('category').cat.codes

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Best hyperparameters
best_params = {
    'subsample': 1.0,
    'reg_lambda': 2,
    'reg_alpha': 0.1,
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.05,
    'gamma': 0.3,
    'colsample_bytree': 0.6,
    'random_state': 42
}

# Initialize and train model
best_model_era = xgb.XGBRegressor(**best_params)
best_model_era.fit(X_train, y_train)

# Predict on test set
y_pred = best_model_era.predict(X_test)

# Evaluate
print("Test R2:", r2_score(y_test, y_pred))


# In[66]:


import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Best hyperparameters for LightGBM
best_params_lgb = {
    'subsample': 0.6,
    'reg_lambda': 1,
    'reg_alpha': 0.1,
    'num_leaves': 100,
    'n_estimators': 300,
    'min_child_samples': 10,
    'max_depth': -1,
    'learning_rate': 0.05,
    'colsample_bytree': 1.0,
    'random_state': 42
}

# Initialize and train model
best_model_lgbm_era = lgb.LGBMRegressor(**best_params_lgb)
best_model_lgbm_era.fit(X_train, y_train)

# Predict on test set
y_pred_lgb = best_model_lgbm_era.predict(X_test)

# Evaluate
print("LightGBM Test R2:", r2_score(y_test, y_pred_lgb))


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Best hyperparameters for Random Forest
best_params_rf = {
    'n_estimators': 300,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': None,
    'bootstrap': False,
    'random_state': 42
}

# Initialize and train model
best_model_rf_era = RandomForestRegressor(**best_params_rf)
best_model_rf_era.fit(X_train, y_train)

# Predict on test set
y_pred_rf = best_model_rf_era.predict(X_test)

# Evaluate
print("Random Forest Test R2:", r2_score(y_test, y_pred_rf))


# In[68]:


# Start from df_sat
df_estimate_era = df_era.copy()
df_estimate_era['date'] = pd.to_datetime(df_estimate_era['date'])
# Start from df_sat
df_estimate_era = df_estimate_era.copy()
# Define bounding box
lon_min, lon_max = 106.56, 107.05
lat_min, lat_max = -6.5, -6.0

# Slice df_merged
df_estimate_era = df_estimate_era[
    (df_estimate_era['longitude'] >= lon_min) &
    (df_estimate_era['longitude'] <= lon_max) &
    (df_estimate_era['latitude'] >= lat_min) &
    (df_estimate_era['latitude'] <= lat_max)
].copy()

# Feature engineering
df_estimate_era['month'] = df_estimate_era['date'].dt.month
df_estimate_era['dayofyear'] = df_estimate_era['date'].dt.dayofyear
df_estimate_era['season'] = df_estimate_era['month'].apply(get_season)
df_estimate_era = df_estimate_era.dropna()


# Drop non-feature columns
drop_cols = ['date', 'date_day', 'station_id']
X_features = df_estimate_era.drop(columns=drop_cols, errors='ignore')

# Encode categorical columns
for col in X_features.select_dtypes(include='object').columns:
    X_features[col] = X_features[col].astype('category').cat.codes

# ---- XGB ----
X_xgb = X_features[best_model_era.get_booster().feature_names]
df_estimate_era['pm25_xgb'] = best_model_era.predict(X_xgb)

# ---- LGBM ----
X_lgbm = X_features[best_model_lgbm_era.feature_name_]
df_estimate_era['pm25_lgbm'] = best_model_lgbm_era.predict(X_lgbm)

# ---- RF ----
X_rf = X_features[best_model_rf_era.feature_names_in_]
df_estimate_era['pm25_rf'] = best_model_rf_era.predict(X_rf)


# In[69]:


df_final_era = df_estimate_era[["longitude", "latitude", "date", "pm25_xgb", "pm25_rf", "pm25_lgbm"]].copy()
df_final = df_estimate[["longitude", "latitude", "date", "pm25_xgb", "pm25_rf", "pm25_lgbm"]].copy()


# In[70]:


# Make a copy
df_estimate_era_clean = df_estimate_era.copy()

# Rename by removing _x and _y suffixes
df_estimate_era_clean.columns = (
    df_estimate_era_clean.columns
    .str.replace(r'_x$', '', regex=True)
    .str.replace(r'_y$', '', regex=True)
)

# If duplicates appear after renaming (e.g., NDVI_x and NDVI_y → NDVI)
# keep the first occurrence
df_estimate_era_clean = df_estimate_era_clean.loc[:, ~df_estimate_era_clean.columns.duplicated()]

df_estimate_era_clean.columns


# In[72]:


# Tambahin kolom year_month
df_final["year_month"] = pd.to_datetime(df_final["date"]).dt.to_period("M")
df_final_era["year_month"] = pd.to_datetime(df_final_era["date"]).dt.to_period("M")

# Create coordinate key
df_final["coord"] = list(zip(df_final["longitude"], df_final["latitude"]))
df_final_era["coord"] = list(zip(df_final_era["longitude"], df_final_era["latitude"]))

# Group by year_month and compare
common_by_month = {}

for ym in df_final["year_month"].unique():
    coords_final = set(df_final.loc[df_final["year_month"] == ym, "coord"])
    coords_era   = set(df_final_era.loc[df_final_era["year_month"] == ym, "coord"])
    
    common_coords = coords_final.intersection(coords_era)
    
    if common_coords:
        common_by_month[str(ym)] = list(common_coords)

# Print summary (show 1st 5 months)
for m, coords in list(common_by_month.items())[:5]:
    print(f"📅 {m}: {len(coords)} common coords")


# In[73]:


import pandas as pd

# Step 1: get all unique coords
unique_coords = df_final[['longitude', 'latitude']].drop_duplicates()

# Step 2: get all unique dates
unique_dates = df_final['date'].drop_duplicates()

# Step 3: make full cartesian product of coords × dates
full_grid = unique_coords.assign(key=1).merge(
    unique_dates.to_frame(name="date").assign(key=1),
    on="key"
).drop("key", axis=1)

# Step 4: merge with df_final to align values
df_completed = pd.merge(
    full_grid,
    df_final,
    on=["longitude", "latitude", "date"],
    how="left"
)

df_completed = df_completed.drop(columns=['coord','year_month'])


# In[74]:


df_merged_ml = pd.merge(
    df_completed,
    df_estimate_era_clean,
    on=["latitude", "longitude", "date"],
    how="left",   # or "inner" if you only want exact matches
    suffixes=("", "_era")
)

print(df_merged_ml)


# In[75]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Features (do not include any pm25_* columns)
features = [
    "t2m", "sp", "tp", "ws", "rh",
    "NDVI", "ndbi", "lst", "elevation", "population",
    "road_density", "built_surface_m2", "avg_rad",
    "month", "dayofyear"
]

# One-hot encode season
X = pd.get_dummies(df_merged_ml[features + ["season"]])
y = df_merged_ml["pm25_xgb"]

# Only train on non-missing rows
mask = y.notna()
X_train, X_val, y_train, y_val = train_test_split(
    X[mask], y[mask], test_size=0.2, random_state=42
)

# Train XGBoost regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Validation
y_pred = xgb_model.predict(X_val)
print("R²:", r2_score(y_val, y_pred))

# Fill missing pm25_xgb
missing_mask = df_merged_ml["pm25_xgb"].isna()
X_missing = pd.get_dummies(df_merged_ml.loc[missing_mask, features + ["season"]])
X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)

df_merged_ml.loc[missing_mask, "pm25_xgb"] = xgb_model.predict(X_missing)


# In[77]:


# Train XGBoost regressor
rf_model = RandomForestRegressor(
    n_estimators= 150,
    min_samples_split= 5,
    min_samples_leaf= 1,
    max_features= 'sqrt',
    max_depth= None,
    bootstrap= False,
    random_state= 42
)
rf_model.fit(X_train, y_train)

# Validation
y_pred = rf_model.predict(X_val)
print("R²:", r2_score(y_val, y_pred))

# Fill missing pm25_xgb
missing_mask = df_merged_ml["pm25_rf"].isna()
X_missing = pd.get_dummies(df_merged_ml.loc[missing_mask, features + ["season"]])
X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)

df_merged_ml.loc[missing_mask, "pm25_rf"] = rf_model.predict(X_missing)


# In[78]:


# Train XGBoost regressor
lgb_model = lgb.LGBMRegressor(
    subsample = 0.1,
    reg_lambda = 1,
    reg_alpha = 2,
    num_leaves = 200,
    n_estimators = 200,
    min_child_samples = 5,
    max_depth = 7,
    learning_rate = 0.1,
    colsample_bytree = 0.5,
    random_state = 42
)
lgb_model.fit(X_train, y_train)

# Validation
y_pred = lgb_model.predict(X_val)
print("R²:", r2_score(y_val, y_pred))

# Fill missing pm25_xgb
missing_mask = df_merged_ml["pm25_lgbm"].isna()
X_missing = pd.get_dummies(df_merged_ml.loc[missing_mask, features + ["season"]])
X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)

df_merged_ml.loc[missing_mask, "pm25_lgbm"] = lgb_model.predict(X_missing)


# In[79]:


df_final_final = df_merged_ml[["longitude", "latitude", "date", "pm25_xgb", "pm25_rf", "pm25_lgbm"]].copy()


# In[81]:


# First, make sure the date columns are datetime
df_pm25_daily["date"] = pd.to_datetime(df_pm25_daily["date"])
df_final_final["date"] = pd.to_datetime(df_final_final["date"])

# Initialize a list to store results
results = []

# Loop through each unique date in PM2.5 data
for dt in df_pm25_daily["date"].unique():
    # Filter PM2.5 and satellite data for this date
    pm25_day = df_pm25_daily[df_pm25_daily["date"] == dt].copy()
    sat_day = df_final_final[df_final_final["date"] == dt].copy()
    
    if sat_day.empty:
        continue  # skip if no satellite data for this date
    
    # Build KDTree for satellite coordinates of this day
    tree = cKDTree(sat_day[["longitude", "latitude"]].values)
    
    # Query nearest satellite point for each PM2.5 station
    pm25_coords = pm25_day[["longitude", "latitude"]].values
    distances, indices = tree.query(pm25_coords, k=1)
    
    # Extract nearest satellite points
    nearest_sat = sat_day.iloc[indices].reset_index(drop=True)
    
    # Merge results
    pm25_day["nearest_sat_lon"] = nearest_sat["longitude"].values
    pm25_day["nearest_sat_lat"] = nearest_sat["latitude"].values
    pm25_day["distance_deg"] = distances
    
    results.append(pm25_day)

# Combine all dates back into a single DataFrame
df_pm25_with_nearest_filtered = pd.concat(results, ignore_index=True)
df_pm25_with_nearest_filtered

# Merge df_pm25_with_nearest_filtered with satellite data
df_merged_final = df_pm25_with_nearest_filtered.merge(
    df_final_final,
    left_on=["date", "nearest_sat_lat", "nearest_sat_lon"],
    right_on=["date", "latitude", "longitude"],
    how="left",
    suffixes=("", "_sat")
)

# Optional: drop redundant latitude/longitude columns from satellite if needed

df_merged_final = df_merged_final.drop(columns=['nearest_sat_lon', 'nearest_sat_lat', 'distance_deg', 'longitude_sat','aqi',
       'latitude_sat'])


# In[83]:


import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Assuming your DataFrame is named df
models = ['pm25_xgb', 'pm25_rf', 'pm25_lgbm']
metrics_dict = {}

for model in models:
    y_true = df_merged_final['pm25']
    y_pred = df_merged_final[model]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    
    metrics_dict[model] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Bias': bias
    }

metrics_df = pd.DataFrame(metrics_dict).T
print(metrics_df)


# In[ ]:


df_final_final.to_csv("pm25_final.csv", index=False)

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
}).reset_index()

# Filter for 2025
df_pm25_daily = df_pm25_daily[
    (pd.to_datetime(df_pm25_daily['date']).dt.year == 2025)
]

from scipy.spatial import cKDTree

# First, make sure the date columns are datetime
df_pm25_daily["date"] = pd.to_datetime(df_pm25_daily["date"])
df_final_final["date"] = pd.to_datetime(df_final_final["date"])

# Initialize a list to store results
results = []

# Loop through each unique date in PM2.5 data
for dt in df_pm25_daily["date"].unique():
    # Filter PM2.5 and satellite data for this date
    pm25_day = df_pm25_daily[df_pm25_daily["date"] == dt].copy()
    sat_day = df_final_final[df_final_final["date"] == dt].copy()
    
    if sat_day.empty:
        continue  # skip if no satellite data for this date
    
    # Build KDTree for satellite coordinates of this day
    tree = cKDTree(sat_day[["longitude", "latitude"]].values)
    
    # Query nearest satellite point for each PM2.5 station
    pm25_coords = pm25_day[["longitude", "latitude"]].values
    distances, indices = tree.query(pm25_coords, k=1)
    
    # Extract nearest satellite points
    nearest_sat = sat_day.iloc[indices].reset_index(drop=True)
    
    # Merge results
    pm25_day["nearest_sat_lon"] = nearest_sat["longitude"].values
    pm25_day["nearest_sat_lat"] = nearest_sat["latitude"].values
    pm25_day["distance_deg"] = distances
    
    results.append(pm25_day)

# Combine all dates back into a single DataFrame
df_pm25_with_nearest_filtered = pd.concat(results, ignore_index=True)
df_pm25_with_nearest_filtered

# Merge df_pm25_with_nearest_filtered with satellite data
df_merged_final = df_pm25_with_nearest_filtered.merge(
    df_final_final,
    left_on=["date", "nearest_sat_lat", "nearest_sat_lon"],
    right_on=["date", "latitude", "longitude"],
    how="left",
    suffixes=("", "_sat")
)

# Optional: drop redundant latitude/longitude columns from satellite if needed

df_merged_final = df_merged_final.drop(columns=['nearest_sat_lon', 'nearest_sat_lat', 'distance_deg', 'longitude_sat','aqi',
       'latitude_sat','aqi'])


# In[14]:


df_merged_final.to_csv("/tif_output/daily_complete.csv")


# In[12]:


import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Ensure longitude, latitude, and date are correct types
df_final_final['date'] = pd.to_datetime(df_final_final['date'])

# Define models to export
models = ['pm25_xgb', 'pm25_rf', 'pm25_lgbm']

# Define output directory
output_dir = "/tif_output/"

# Get grid boundaries
lon_unique = np.sort(df_final_final['longitude'].unique())
lat_unique = np.sort(df_final_final['latitude'].unique())

# Calculate resolution (assumes regular grid)
res_lon = np.abs(lon_unique[1] - lon_unique[0])
res_lat = np.abs(lat_unique[1] - lat_unique[0])

# Create transform for rasterio
transform = from_origin(west=lon_unique.min() - res_lon/2,
                        north=lat_unique.max() + res_lat/2,
                        xsize=res_lon,
                        ysize=res_lat)

# Loop through models and dates
for model in models:
    for date, group in df_final_final.groupby('date'):
        # Pivot table into grid form
        pivot = group.pivot_table(index='latitude', columns='longitude', values=model)
        
        # Ensure rows are top-down (north to south)
        data = np.flipud(pivot.values)
        
        # Define output path
        date_str = date.strftime('%Y-%m-%d')
        out_path = f"{output_dir}{model}_{date_str}.tif"

        # Write to GeoTIFF
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=str(data.dtype),
            crs="EPSG:4326",  # WGS84 lat/lon
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        print(f"✅ Saved {out_path}")

