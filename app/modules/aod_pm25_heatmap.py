import streamlit as st
from utils.db import get_connection
import pandas as pd
import os
import psycopg2
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.spatial import cKDTree
from folium import CircleMarker
from folium.plugins import MiniMap
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error,r2_score
from streamlit_option_menu import option_menu
from datetime import datetime
from google import genai
from google.genai import types

# --- Initialize Gemini Client ---
client = genai.Client(vertexai=True, project="double-reef-468803-r9", location="us-central1")

def show():
	st.markdown(
		"""
		<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
			PM2.5 Prediction Heatmap
		</div>
		""",
		unsafe_allow_html=True
	)

	cssabout = """
	.st-key-about_aod {
		background-color: white;
		padding: 20px;
		border-radius: 10px;
		margin-bottom: 20px;
	}
	"""
	st.html(f"<style>{cssabout}</style>")		

	with st.container(key="about_aod"):

		st.markdown("""
			<div style="font-size:16px; font-weight:500; margin-bottom:10px;">
				PM2.5 Prediction Based on Machine Learning Models Using Remote-sensing Data in Jakarta
			</div>

			<div style="font-size:14px; font-weight:300; margin-bottom:10px;text-align:justify;">
				
			One of the fine airborne pollutant is **PM2.5** with a diameter less than 2.5 µm, means they **efficiently penetrate the human respiratory system**. 
			Despite continuous PM2.5 monitoring by government and international agencies, **Jakarta’s unevenly distributed stations** limit spatially consistent mapping of PM2.5. 
			Several studies have utilized machine learning models such as **Random Forest (RF), XGBoost, and LightGBM** to predict PM2.5 concentrations 

			</div>

			<div style="font-size:14px; font-weight:300; margin-bottom:10px;text-align:justify;">
			Predicting the PM2.5 concentration is necessary for social planning and environmental management, to mitigate the impact of air pollution on public health. 
			This project focuses on predicting PM2.5 concentrations in Jakarta to cover the gaps caused by the city’s scattered air quality network using machine learning models (Random Forest, XGBoost, and LightGBM).			
			</div>
			""", unsafe_allow_html=True)
			
		st.markdown("<br>", unsafe_allow_html=True)

	css = """
		.st-key-selector_box,.st-key-analysis, .st-key-map,.st-key-table {
			background-color: white;
			padding: 20px;
			border-radius: 10px;
			margin-bottom: 20px;
		}
		"""
	st.html(f"<style>{css}</style>")

	with st.container(key="selector_box"): 
		import pandas as pd
		import os

		# --- Input folders ---
		tif_dir = "/app/tif_output/"
		models = ["xgb", "rf", "lgbm"]

		# --- Load daily station dataframe ---
		df_pm25_daily = pd.read_csv("/app/tif_output/daily_complete.csv", parse_dates=["date"])
		df_pm25_daily = df_pm25_daily[df_pm25_daily["pm25"] != 0]

		selected_model = st.selectbox("Select Model", models)
		# Collect TIFFs for selected model
		tifs = [
			f for f in os.listdir(tif_dir)
			if f.endswith(".tif") and f"_{selected_model}_" in f
		]
		if not tifs:
			st.error(f"No TIFF files found for model {selected_model}")
			st.stop()

		# Extract available dates from filenames
		date_options = [f.split("_")[-1].replace(".tif", "") for f in tifs]
		date_options = ["All Dates"] + sorted(date_options)  # prepend "All Dates"
		selected_date = st.selectbox("Select Date", date_options)
		
	
	# 🔘 Selectors with custom container
	with st.container(key="map"): 
		map_col, space_col, scatter_col = st.columns([2.5, 0.2, 1.6])

		with map_col:
			import folium
			from streamlit_folium import st_folium
			from folium.plugins import MiniMap
			import rasterio
			from rasterio.warp import reproject, Resampling
			import numpy as np
			from matplotlib import colors, cm
			from branca.colormap import LinearColormap			

			# --- Create Folium Map ---
			m = folium.Map(location=[-6.2, 106.8], zoom_start=11, tiles="cartodbpositron")
			
			vmin, vmax = 0, 80
			cmap = cm.turbo
			norm = colors.Normalize(vmin=vmin, vmax=vmax)

			# --- Load raster ---
			if selected_date == "All Dates":
				raster_stack = []
				reference_meta = None
				reference_shape = None

				for i, tif_file in enumerate(tifs):
					with rasterio.open(os.path.join(tif_dir, tif_file)) as src_temp:
						img_temp = src_temp.read(1).astype(float)
						if src_temp.nodata is not None:
							img_temp = np.where(img_temp == src_temp.nodata, np.nan, img_temp)

						if i == 0:
							# Use the first file as reference
							reference_meta = src_temp.meta.copy()
							reference_shape = img_temp.shape
							bounds = src_temp.bounds
							raster_stack.append(img_temp)
						else:
							# Resample to match reference grid
							img_resampled = np.empty(reference_shape, dtype=float)
							rasterio.warp.reproject(
								source=img_temp,
								destination=img_resampled,
								src_transform=src_temp.transform,
								src_crs=src_temp.crs,
								dst_transform=reference_meta["transform"],
								dst_crs=reference_meta["crs"],
								resampling=Resampling.bilinear,
							)
							raster_stack.append(img_resampled)

				image = np.nanmean(raster_stack, axis=0)
			else:
				tif_file = [f for f in tifs if selected_date in f][0]
				tif_path = os.path.join(tif_dir, tif_file)
				with rasterio.open(tif_path) as src:
					bounds = src.bounds
					image = src.read(1)
					nodata = src.nodata
					if nodata is not None:
						image = np.where(image == nodata, np.nan, image)

			# --- Prepare raster for folium overlay ---
			image_masked = np.ma.masked_invalid(image)
			rgba_img = (cmap(norm(image_masked.filled(np.nan))) * 255).astype(np.uint8)

			folium.raster_layers.ImageOverlay(
				image=rgba_img,
				bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
				opacity=0.75,
				name="PM2.5 Raster"
			).add_to(m)

			# --- Prepare station data ---
			if selected_date == "All Dates":
				# Aggregate observed PM2.5 for all dates
				stations_today = (
					df_pm25_daily
					.groupby(["station", "latitude", "longitude"], as_index=False)
					.agg({"pm25": "mean", f"pm25_{selected_model}": "mean"})
				)
			else:
				stations_today = df_pm25_daily[df_pm25_daily["date"] == pd.to_datetime(selected_date)][
					["station", "latitude", "longitude", "pm25", f"pm25_{selected_model}"]
				]

			# --- Overlay stations and compute errors ---
			errors = []
			for _, row in stations_today.iterrows():
				pm25_station = row["pm25"]

				# raster index (use bounds from one raster, assume consistent grid)
				with rasterio.open(os.path.join(tif_dir, tifs[0])) as src_ref:
					row_idx, col_idx = src_ref.index(row["longitude"], row["latitude"])

				if (0 <= row_idx < image.shape[0]) and (0 <= col_idx < image.shape[1]):
					pm25_raster = image[row_idx, col_idx]
				else:
					pm25_raster = np.nan

				error = pm25_station - pm25_raster
				errors.append({
					"latitude": row["latitude"],
					"longitude": row["longitude"],
					"station_val": pm25_station,
					"raster_val": pm25_raster,
					"error": error
				})

				color = colors.to_hex(cmap(norm(pm25_station)))
				folium.CircleMarker(
					location=[row["latitude"], row["longitude"]],
					radius=5,
					color="black",
					weight=1,
					fill=True,
					fill_color=color,
					fill_opacity=0.75,
					popup=folium.Popup(
						html=(f'<div style="font-size:12px;">'
							f"<b>Station PM2.5:</b> {pm25_station:.1f}<br>"
							f"<b>Raster PM2.5:</b> {pm25_raster:.1f}<br>"
							f"<b>Error:</b> {error:.1f}"
							f'</div>'),
						max_width=200
					)
				).add_to(m)

			# --- Add colormap legend ---
			colormap = LinearColormap(
				colors=[colors.to_hex(cmap(norm(v))) for v in np.linspace(vmin, vmax, 256)],
				vmin=vmin, vmax=vmax
			)

			colormap.add_to(m)
			
			carto_tiles = folium.TileLayer(
				tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
				attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
				name="CartoDB Light",
				subdomains="abcd",
				max_zoom=19
				)

			#minimap = MiniMap(tile_layer=carto_tiles, toggle_display=True, position="bottomright")
			#m.add_child(minimap)

			folium.LayerControl().add_to(m)

			st.markdown(f"""
								<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">
									Station vs Estimated PM2.5 Map
								</div>
							""", unsafe_allow_html=True)

			# --- Show map ---
			st_map = st_folium(m, use_container_width=True, height=500)


		with scatter_col:
			import pandas as pd
			import numpy as np
			import altair as alt
			from sklearn.metrics import r2_score, mean_absolute_error

			# --- Load data ---
			df_all = pd.read_csv("/app/tif_output/daily_complete.csv")
			df_all = df_all.loc[:, ~df_all.columns.str.contains("Unnamed")]
			df_all = df_all[df_all["pm25"] != 0]

			# --- Standardize column names ---
			df_all = df_all.rename(columns={"pm25": "station_val"})

			# --- Ensure 'date' is datetime.date type ---
			df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.date

			# --- Raster column based on selected model ---
			raster_col = f"pm25_{selected_model}"

			# --- Filter or aggregate ---
			if selected_date == "All Dates":
				df_errors = (
					df_all.groupby("station", as_index=False)
					.agg({"station_val": "mean", raster_col: "mean"})
				)
			else:
				# Convert selected_date to date object
				selected_date_obj = pd.to_datetime(selected_date).date()

				# Filter by selected date
				df_errors = df_all[df_all["date"] == selected_date_obj].copy()

			# --- Compute error ---
			if not df_errors.empty:
				df_errors["error"] = df_errors["station_val"] - df_errors[raster_col]

				# --- Display ---
				st.markdown(f"""
					<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">
						Scatter Station vs Estimated PM2.5
					</div>
				""", unsafe_allow_html=True)

				df_errors = df_errors.dropna(subset=["station_val", raster_col])

				obs_col = "station_val"
				pred_col = raster_col
				obs_display = "Station PM2.5 (µg/m³)"
				pred_display = f"Estimated PM2.5 ({selected_model.upper()})"

				obs = df_errors[obs_col]
				pred = df_errors[pred_col]

				# --- Metrics ---
				R2 = r2_score(obs, pred)
				MAE = mean_absolute_error(obs, pred)
				RMSE = np.sqrt(np.mean((pred - obs) ** 2))  # ✅ fixed
				Bias = np.mean(pred - obs)
				RPE = np.mean(np.abs(pred - obs) / obs) * 100

				# --- Scatter plot ---
				vmin, vmax = df_errors[[obs_col, pred_col]].min().min(), df_errors[[obs_col, pred_col]].max().max()

				scatter = (
					alt.Chart(df_errors)
					.mark_circle(size=80, opacity=0.7)
					.encode(
						x=alt.X(obs_col, title=obs_display),
						y=alt.Y(pred_col, title=pred_display),
						color=alt.Color(
							"error",
							scale=alt.Scale(
								domain=[df_errors["error"].min(), df_errors["error"].max()],
								range=["blue", "green", "yellow", "orange", "red"]  # custom gradient
							),
							legend=alt.Legend(
								title="Error",
								orient="right",
								gradientLength=380  # ⬅️ increase this for taller colorbar
							)
						),
						tooltip=[alt.Tooltip(c, title=c) for c in df_errors.columns]
					)
					.properties(width=250, height=500)
					.interactive()
				)

				# --- 1:1 reference line ---
				ref_line = (
					alt.Chart(pd.DataFrame({"x": [vmin, vmax], "y": [vmin, vmax]}))
					.mark_line(color="black", strokeDash=[4, 4])
					.encode(x="x", y="y")
				)

				# --- Metrics overlay ---
				metrics_text = pd.DataFrame({
					"metrics": [f"R²: {R2:.2f}", f"MAE: {MAE:.2f}", f"RMSE: {RMSE:.2f}",
								f"Bias: {Bias:.2f}", f"RPE: {RPE:.2f}%"],
					"x": [vmax]*5,
					"y": [vmin + (i+1)*(vmax-vmin)*0.03 for i in range(5)]
				})

				metrics = (
					alt.Chart(metrics_text)
					.mark_text(align='right', fontSize=10, fontWeight='bold', color='black')
					.encode(
						x=alt.X('x:Q'),  # Quantitative
						y=alt.Y('y:Q'),  # Quantitative
						text=alt.Text('metrics:N')  # Nominal text
					)
				)

				st.altair_chart(scatter + ref_line + metrics, use_container_width=True)

			else:
				st.warning("⚠️ No data available for the selected date.")

	errors_df = pd.DataFrame(errors)

	# --- Assign zones based on coordinates ---
	def assign_zones(errors_df):
		conditions = {
			"North": errors_df["latitude"] > -6.1,
			"Central": (errors_df["latitude"].between(-6.2, -6.1)) & (errors_df["longitude"].between(106.78, 106.85)),
			"West": errors_df["longitude"] < 106.75,
			"East": errors_df["longitude"] > 106.9,
			"South": errors_df["latitude"] < -6.25
		}
		zone_col = []
		for i in range(len(errors_df)):
			assigned = [z for z, cond in conditions.items() if cond.iloc[i]]
			zone_col.append(assigned[0] if assigned else "Unknown")
		errors_df["zone"] = zone_col
		return errors_df

	errors_df = assign_zones(errors_df)

	# --- Display container for AI analysis ---
	with st.container(key="analysis"):
		zone_summary = (
			errors_df.groupby("zone")
			.agg({
				"station_val": "mean",
				"raster_val": "mean",
				"error": ["mean", "std"]
			})
			.round(3)
		)
		zone_summary.columns = ["Station Mean", "Model Mean", "Mean Error", "Error Std"]

		# --- Example AI summary template ---
		example_summary = """
		- **Overall Model:** R² = 0.83 indicates good correlation, though underestimation occurs in the South.
		- **Spatial Pattern:** Central and West zones show higher bias, possibly due to coarse urban emission estimates.
		- **Notes:** High estimated PM2.5 in East Jakarta possibly due to Industrial activity near Bekasi and Karawang.
		"""

		# --- Gemini Prompt ---
		prompt = f"""
		You are an environmental data analyst. Analyze the spatiotemporal results for PM2.5 model performance in Jakarta, do not show the data directly
		**Background Theories for PM2.5 in Jakarta**

		**1. Seasonal Behavior**
		Jakarta has a tropical monsoon climate with two main seasons:
		- **Wet Season (November–March):** Frequent rainfall and stronger winds lower PM2.5 through wet deposition and atmospheric cleansing.
		- **Transition Periods (April–May, October):** Variable conditions can lead to fluctuating PM2.5 levels depending on rainfall frequency and local emissions.
		- **Dry Season (June–September):** Reduced rainfall, calm winds, and frequent temperature inversions trap pollutants near the surface. Biomass burning and stagnant air often cause **higher PM2.5** concentrations.

		➡️ **In summary:**
		- **June–September:** higher PM2.5 (dry, stagnant, and hazy conditions)
		- **November–March:** lower PM2.5 (wet, cleaner air)
		- **April–May and October:** transitional, moderate to fluctuating PM2.5

		Use this knowledge to interpret whether the observed PM2.5 level on the selected date aligns with seasonal expectations.

		---

		**2. Spatial Characteristics of Jakarta**
		Jakarta’s PM2.5 levels vary spatially due to differences in land use, emission sources, and airflow patterns:
		- **Central Jakarta:** Dense business and traffic areas (CBD) — high vehicle emissions, leading to consistently higher PM2.5 levels.  
		- **West Jakarta:** Mixed residential and industrial zones, often with moderate PM2.5 influenced by local activities.  
		- **East Jakarta:** Proximity to **industrial areas in Bekasi and Karawang**, frequently experiences **elevated PM2.5** levels, especially during calm conditions.  
		- **North Jakarta:** Coastal and port-related activities (e.g., Tanjung Priok) contribute to emissions but can experience dilution from sea breezes.  
		- **South Jakarta:** More greenery and open spaces (e.g., residential, parks) — tends to record **lower PM2.5**, though can still be affected by regional transport from north or east.  

		Use this context when interpreting **zone-level biases and spatial variability** in model performance.

		**Date**
		{selected_date}

		**Machine Learning model used**
		{selected_model}

		**Evaluation Metrics**
		- R² = {R2:.3f}
		- MAE = {MAE:.3f}
		- RMSE = {RMSE:.3f}
		- Bias = {Bias:.3f}
		- RPE = {RPE:.2f}%

		**Jakarta Zone**
		{zone_summary.to_markdown()}

		**Sample Data:**
		{errors_df.to_csv(index=False)}

		**Example Format**
		{example_summary}

		**Tasks**
		1. Provide a concise overall spatial and temporal(seasonality based on dates) and performance summary (under 150 words).
		2. Identify which Jakarta city (zones) show high bias or variability and analyze the PM2.5 spatial pattern.
		3. Explain model behavior possible environmental or model causes (topography, urban sources, etc.).
		4. Output in **bullet points** with short paragraphs highlight with bold for important statements.
		"""
		with st.spinner("Analyzing PM2.5 spatial performance across Jakarta..."):
			try:
				response = client.models.generate_content(
					model="gemini-2.0-flash",
					contents=prompt
				)

				st.markdown(f"""
				<div style="
					padding:14px;
					border-radius:10px;
					margin-top:5px;
					margin-bottom:10px;
					font-size:13px;
					font-weight:300;
					text-align:justify;
					background-color:#f8f9fa;
					border:1px solid #e0e0e0;">
					{response.text}</div>
					""", unsafe_allow_html=True)
			except Exception as e:
				st.error(f"Error calling Gemini AI: {e}")

	with st.container(key="table"): 
	# --- Show error table and scatter ---
		if selected_date == "All Dates":
			st.markdown(f"""
							<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">
								Station vs Estimated Errors (All Dates)
							</div>
						""", unsafe_allow_html=True)
		else:
			st.markdown(f"""
							<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">
								Station vs Raster Errors ({selected_date})
							</div>
						""", unsafe_allow_html=True)

			# --- Display the table with nice column headers ---
		display_columns = {
					"station": "Station",
					"station_val": "Station PM2.5",
					raster_col: f"Estimated PM2.5 ({selected_model})",
					"error": "Error"}

		st.dataframe(df_errors.rename(columns=display_columns))

		st.markdown("<br>", unsafe_allow_html=True)

		