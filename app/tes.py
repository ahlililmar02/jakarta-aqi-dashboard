import streamlit as st
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

st.markdown(
	"""
	<style>
	/* Scale main page content */
	.main .block-container {
		transform: scale(0.8);
		transform-origin: top left;
		width: 125%;  /* compensate for the scale */
	}

	/* Scale sidebar */
	.sidebar .sidebar-content {
		transform: scale(0.8);
		transform-origin: top left;
		width: 125%;  /* compensate for the scale */
	}
	</style>
	""",
	unsafe_allow_html=True
)

# Set wide layout
st.set_page_config(
	page_title="Air Quality Dashboard",
	 page_icon="🌫️",
	layout="wide",
	initial_sidebar_state="auto",
	menu_items={
		'Report a bug': "https://github.com/ahlililmar02/AQIDashboard/issues",  # proper bug reporting
	}
)

st.markdown(
	"""
	<style>
	/* Sidebar title font */
	.sidebar .title {
		font-size: 18px !important;  /* adjust as needed */
	}

	/* Option menu font */
	.sidebar .nav-link {
		font-size: 14px !important;  /* adjust menu item font size */
	}

	/* Optional: reduce icons size in option_menu */
	.sidebar .nav-link svg {
		width: 16px;
		height: 16px;
	}
	</style>
	""",
	unsafe_allow_html=True
)

with st.sidebar:
	st.title("Jakarta Air Quality Dashboard")  # Title without icon

	page = option_menu(
		menu_title=None,
		options=["Air Quality Monitor", "Download Data", "AOD Derived PM2.5 Heatmap", "About"],
		icons=["bar-chart", "download", "cloud", "info-circle"],
		default_index=0
	)

	# Social links at the bottom
	st.markdown(
		"""
		<div style='position: fixed; bottom: 10px;'>
			<a href='mailto:ahlililmar02@gmail.com' target='_blank' style='text-decoration:none;'>
				<img src='https://cdn.jsdelivr.net/npm/simple-icons@v10/icons/gmail.svg' width='25' style='vertical-align:middle;margin-right:10px;'/>
			</a>
			<a href='https://www.linkedin.com/in/ahlil-batuparan-850b6b243/' target='_blank' style='text-decoration:none;'>
				<img src='https://cdn.jsdelivr.net/npm/simple-icons@v10/icons/linkedin.svg' width='25' style='vertical-align:middle;margin-right:10px;'/>
			</a>
			<a href='https://github.com/ahlililmar02/' target='_blank' style='text-decoration:none;'>
				<img src='https://cdn.jsdelivr.net/npm/simple-icons@v10/icons/github.svg' width='25' style='vertical-align:middle;margin-right:10px;'/>
			</a>
		</div>
		""",
		unsafe_allow_html=True
	)

st.markdown("""
<style>
/* Main page background */
body, .stApp {
	background-color: #f0f0f0;  /* page background color */
	color: #111111;             /* default text color */
}
</style>
""", unsafe_allow_html=True)


def get_connection():
	return psycopg2.connect(
		host=os.environ.get("DB_HOST"),
		port=os.environ.get("DB_PORT"),
		dbname=os.environ.get("DB_NAME"),
		user=os.environ.get("DB_USER"),
		password=os.environ.get("DB_PASS")
	)


with open("static/style.css") as f:
	st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#  PAGE 1: DASHBOARD
if page == "Air Quality Monitor":
	st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								Real-Time Air Quality Dashboard
							</div>
						""", unsafe_allow_html=True)
						
	css3 = """
	.st-key-about {
		background-color: white;
		padding: 20px;
		border-radius: 10px;
		margin-bottom: 20px;
	}
	"""

	st.html(f"<style>{css3}</style>")

	with st.container(key="about"):
			st.markdown("""
			<div style="font-size:16px; font-weight:500; margin-bottom:10px;">
					What is Air Quality Index (AQI)?
				</div>
			   
			<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
			Air Quality Index (AQI) is an indicator used to communicate how polluted the air currently is, and what associated health effects might be a concern for you. The AQI focuses on health effects you may experience within a few hours or days after breathing polluted air. Here's how to interpret the AQI values:
			</div>
			<style>
				.aqi-table {
					border-collapse: collapse;
					width: 100%;
					font-size: 12px;
				}
				.aqi-table th, .aqi-table td {
					border: 1px solid #ddd;
					padding: 8px;
					text-align: center;
				}
				.aqi-table th {
					background-color: #f2f2f2;
				}
			</style>

			<table class="aqi-table">
			<tr>
				<th>AQI Range</th>
				<th>PM2.5 (µg/m³)</th>
				<th>Level of Health Concern</th>
			</tr>
			<tr>
				<td>0 – 50</td>
				<td>0.0 – 12.0</td>
				<td style="background-color:#66c2a4;">Good</td>
			</tr>
			<tr>
				<td>51 – 100</td>
				<td>12.1 – 35.4</td>
				<td style="background-color:#ffe066;">Moderate</td>
			</tr>
			<tr>
				<td>101 – 150</td>
				<td>35.5 – 55.4</td>
				<td style="background-color:#ffb266;">Unhealthy for Sensitive Groups</td>
			</tr>
			<tr>
				<td>151 – 200</td>
				<td>55.5 – 150.4</td>
				<td style="background-color:#ff6666;">Unhealthy</td>
			</tr>
			<tr>
				<td>201 – 300</td>
				<td>150.5 – 250.4</td>
				<td style="background-color:#b266ff;">Very Unhealthy</td>
			</tr>
			<tr>
				<td>301+</td>
				<td>250.5+</td>
				<td style="background-color:#d2798f;">Hazardous</td>
			</tr>
			</table>
			""", unsafe_allow_html=True)

	# Filter to only today's data
	# Connect and read data
	@st.cache_data(ttl=3600) 
	def load_data(today=None):
		# Use today's date if not provided
		if today is None:
			today = datetime.now().strftime('%Y-%m-%d')

		conn = get_connection()

		# Only select needed columns and filter to today
		query = f"""
			SELECT station, sourceid, time, aqi, pm25, latitude, longitude
			FROM aqi
			WHERE time >= '{today} 00:00:00'
			AND aqi IS NOT NULL
			AND aqi != 0
		"""

		df_today = pd.read_sql(query, conn)
		conn.close()
		return df_today
	df_today = load_data()


	# ✅ Get latest data per station *from today's data only*
	df_latest = df_today.sort_values("time").groupby("station", as_index=False).last()

	# ✅ Add color
	def get_rgba_color(aqi, alpha=0.7):
		if pd.isna(aqi): return f"rgba(200, 200, 200, {alpha})"
		elif aqi <= 50: return f"rgba(0, 228, 0, {alpha})"
		elif aqi <= 100: return f"rgba(255, 255, 0, {alpha})"
		elif aqi <= 150: return f"rgba(255, 126, 0, {alpha})"
		elif aqi <= 200: return f"rgba(255, 0, 0, {alpha})"
		elif aqi <= 300: return f"rgba(143, 63, 151, {alpha})"
		else: return f"rgba(126, 0, 35, {alpha})"

	df_latest["color"] = df_latest["aqi"].apply(get_rgba_color)

	css = """
	.st-key-selector_box {
		background-color: white;
		padding: 20px;
		border-radius: 10px;
		margin-bottom: 20px;
	}
	"""
	st.html(f"<style>{css}</style>")

	# 🔘 Selectors with custom container
	with st.container(key="selector_box"):
		sourceid_list = df_latest["sourceid"].unique()

		# Set default for source ID (e.g., first one or a specific value)
		default_source = sourceid_list[0]  # or 'SOME_SOURCE_ID' if you know the ID
		selected_source = st.selectbox("Select Source ID", sourceid_list, index=list(sourceid_list).index(default_source))

		stations_in_source = df_latest[df_latest["sourceid"] == selected_source]["station"].unique()

		# Set default for station (e.g., first one or a specific station)
		default_station = stations_in_source[0]
		selected_station = st.selectbox("Select Station", stations_in_source, index=list(stations_in_source).index(default_station))

		selected_row = df_latest[df_latest["station"] == selected_station].iloc[0]
		center = [selected_row["latitude"], selected_row["longitude"]]


	st.markdown("<br>", unsafe_allow_html=True)

	# 🗺️ Build folium map
	m = folium.Map(
		location=center,
		zoom_start=11,
		control_scale=True,
		scrollWheelZoom=True,
		tiles="CartoDB positron",
	)

	carto_tiles = folium.TileLayer(
		tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
		attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
		name="CartoDB Light",
		subdomains="abcd",
		max_zoom=19
		)

	minimap = MiniMap(tile_layer=carto_tiles, toggle_display=True, position="bottomright")
	m.add_child(minimap)

	from folium.features import DivIcon

	# Filter hanya data dari selected_source
	filtered_df = df_latest[df_latest["sourceid"] == selected_source]

	# Loop untuk semua station dalam source itu
	for _, row in filtered_df.iterrows():
		aqi = row["aqi"]
		color = row["color"]
		label = f"{int(aqi)}" if pd.notna(aqi) else "?"

		is_selected = row["station"] == selected_station
		size = 28 if is_selected else 24
		font_size = "11px" if is_selected else "10px"
		border = "2px solid white" if is_selected else "none"

		folium.Marker(
			location=[row["latitude"], row["longitude"]],
			icon=DivIcon(
				icon_size=(size, size),
				icon_anchor=(size // 2, size // 2),
				html=f"""
				<div style='
					background-color:{color};
					color:white;
					font-size:{font_size};
					font-weight:bold;
					border-radius:50%;
					width:{size}px;
					height:{size}px;
					text-align:center;
					line-height:{size}px;
					box-shadow: 0 0 2px #333;
					border:{border};'>
					{label}
				</div>
				""",
			),
			tooltip=f"{row['station']}",
			popup=folium.Popup(
				f"""
				<div style='font-size: 13px; line-height: 1.5'>
					<b>Station:</b> {row['station']}<br/>
					<b>Latest Time:</b> {row['time'].strftime('%Y-%m-%d %H:%M')}<br/>
					<b>AQI:</b> {row['aqi']:.0f}<br/>
					<b>PM2.5:</b> {row['pm25']:.1f} µg/m³
				</div>
				""",
				max_width=500,
			),
		).add_to(m)

	# 🌍 Show map full-width
	css2 = """
	.st-key-map {
		background-color: white;
		padding: 20px;
		border-radius: 10px;
		margin-bottom: 20px;
	}
	"""
	st.html(f"<style>{css2}</style>")

	with st.container(key="map"):
		
		map_output = st_folium(m, height=500,use_container_width=True, returned_objects=["last_object_clicked"])

		# 📘 Legend
		legend_html = """
		<div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px;">
			<div style="display: flex; align-items: center; gap: 4px;">
				<div style="background-color: rgba(0, 228, 0, 0.7); width: 12px; height: 12px; border: 1px solid #000;"></div> Good (0–50)
			</div>
			<div style="display: flex; align-items: center; gap: 4px;">
				<div style="background-color: rgba(255, 255, 0, 0.7); width: 12px; height: 12px; border: 1px solid #000;"></div> Moderate (51–100)
			</div>
			<div style="display: flex; align-items: center; gap: 4px;">
				<div style="background-color: rgba(255, 126, 0, 0.7); width: 12px; height: 12px; border: 1px solid #000;"></div> Unhealthy for SG (101–150)
			</div>
			<div style="display: flex; align-items: center; gap: 4px;">
				<div style="background-color: rgba(255, 0, 0, 0.7); width: 12px; height: 12px; border: 1px solid #000;"></div> Unhealthy (151–200)
			</div>
			<div style="display: flex; align-items: center; gap: 4px;">
				<div style="background-color: rgba(143, 63, 151, 0.7); width: 12px; height: 12px; border: 1px solid #000;"></div> Very Unhealthy (201–300)
			</div>
			<div style="display: flex; align-items: center; gap: 4px;">
				<div style="background-color: rgba(126, 0, 35, 0.7); width: 12px; height: 12px; border: 1px solid #000;"></div> Hazardous (301+)
			</div>
		</div>
		"""
		st.markdown(legend_html, unsafe_allow_html=True)
		st.markdown("<br>", unsafe_allow_html=True)

	if map_output and map_output["last_object_clicked"]:
		lat_click = map_output["last_object_clicked"]["lat"]
		lon_click = map_output["last_object_clicked"]["lng"]
		df_latest["distance"] = ((df_latest["latitude"] - lat_click)**2 + (df_latest["longitude"] - lon_click)**2)
		nearest_station = df_latest.sort_values("distance").iloc[0]["station"]
		selected_station = nearest_station
		st.success(f"📌 Selected from map: {selected_station}")

	# 7. Now compute station data (based on final selected_station)
	station_df = df_today[df_today["station"] == selected_station].sort_values("time")

	latest_row = station_df.iloc[-1]

	# 8. Split into 3 columns: left = metrics + chart, middle = space, right = top 5 AQI
	left_col, middle_col, right_col = st.columns([2.5, 0.01, 1.8])

	st.html("""
	<style>
	.st-key-left_box, .st-key-right_box,.st-key-right_box_low, .st-key-time_series, .st-key-bar_chart {
		background-color: white;
		padding: 16px 16px;
		border-radius: 8px;
		margin-bottom: 7px;
	}
	</style>
	""")

	with st.container():
		with left_col:
			
			with st.container(key="left_box"):
					st.markdown(f"""
						<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">
							Latest from {latest_row["station"]}
						</div>
			
						<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
							Here are the metrics of the station's most recent available data.
						</div>
						
					""", unsafe_allow_html=True)
					# 📊 Scorecards
					st.markdown("<br>", unsafe_allow_html=True)
					col1, col2, col3 = st.columns(3)

					# Styling values
					time_value = latest_row["time"].strftime('%H:%M')
					aqi_value = latest_row["aqi"]
					pm_value = latest_row["pm25"]
					color = get_rgba_color(aqi_value)

					def card_style(label, value, color="#ffffff"):
						return f"""
							<div style="
								background-color:{color};
								padding:14px;
								border-radius:10px;
								margin-bottom: 5px;
								text-align:center;
							">
								<p style='font-size:14px;margin:0;'>{label}</p>
								<p style='font-size:14px;margin:0;'>{value}</p>
							</div>
						"""

					# Column 1: Time
					col1.markdown(card_style(label="Time", value=time_value), unsafe_allow_html=True)

					# Column 2: AQI with color
					col2.markdown(card_style(label="AQI", value=f"{aqi_value:.0f}", color=color), unsafe_allow_html=True)

					# Column 3: PM2.5
					col3.markdown(card_style(label="PM2.5", value=f"{pm_value:.1f} µg/m³"), unsafe_allow_html=True)
					st.markdown("<br>", unsafe_allow_html=True)

			
			with st.container(key="time_series"):
					import altair as alt

					# 📈 Time series
					st.markdown("""
					<div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">
						Time Series
					</div>

					<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
							Hourly PM2.5 and AQI time series for today
					</div>
					""", unsafe_allow_html=True)
					st.markdown("<br>", unsafe_allow_html=True)

					# Prepare DataFrame
					plot_station_df = station_df.rename(columns={"pm25": "PM2_5"})

					# Melt PM2_5 and aqi into long format
					plot_station_df = plot_station_df.melt(
						id_vars=["time"],          # keep time as identifier
						value_vars=["PM2_5", "aqi"],  # columns to melt
						var_name="Metric",
						value_name="Value"
					)

					# Optional: convert types to avoid Altair errors
					plot_station_df["Value"] = plot_station_df["Value"].astype(float)
					plot_station_df["time"] = pd.to_datetime(plot_station_df["time"])

					# Optional: custom colors
					color_scale = alt.Scale(domain=["PM2_5", "aqi"], range=["#1f77b4", "#87CEFA"])

					# Create chart
					chart = (
						alt.Chart(plot_station_df)
						.mark_line()
						.encode(
							x=alt.X("time:T", title="Time"),
							y=alt.Y("Value:Q", title="Value (µg/m³ or AQI)"),
							color=alt.Color("Metric:N", scale=color_scale),
							tooltip=["time", "Metric", "Value"]
						)
						.properties(width=700, height=250)
						.configure_view(fill='white')  # chart background white
					)

					st.altair_chart(chart, use_container_width=True)

			from datetime import datetime, timedelta

			today = datetime.now()
			
			def load_weekly_data(start_of_week, end_of_week):
				conn = get_connection()

				query = f"""
					SELECT station, sourceid, time, aqi, pm25, latitude, longitude
					FROM aqi
					WHERE time BETWEEN '{start_of_week}' AND '{end_of_week}'
					AND aqi IS NOT NULL
					AND aqi != 0
				"""

				df_week = pd.read_sql(query, conn)
				conn.close()
				return df_week

			# Start of week as datetime
			start_of_week = today - timedelta(days=today.weekday())  
			end_of_week   = start_of_week + timedelta(days=6)        

			# Format as strings for SQL
			start_of_week_str = start_of_week.strftime('%Y-%m-%d 00:00:00')
			end_of_week_str   = end_of_week.strftime('%Y-%m-%d 23:59:59')

			# Now pass start_of_week_str and end_of_week_str to your SQL query
			df_week = load_weekly_data(start_of_week_str, end_of_week_str)

			# 📈 Display as bar chart
			with st.container(key="bar_chart"):
				st.markdown("""
					<div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">
						AQI and PM2.5 This Week
					</div>

					<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
						Daily average of PM2.5 and AQI bar chart for this week
					</div>
				""", unsafe_allow_html=True)
				
				import altair as alt
				
				weekly_df = df_week[df_week["station"] == selected_station].sort_values("time")

				# 1️⃣ Extract date (no hour)
				weekly_df["date"] = weekly_df["time"].dt.date  # strips hour

				# 2️⃣ Group by that date
				daily_avg = weekly_df.groupby("date")[["aqi", "pm25"]].mean().reset_index()

				# 3️⃣ Optional: Rename column for PM2.5
				daily_avg.rename(columns={"pm25": "PM2_5"}, inplace=True)

				# Ensure 'date' is a datetime, then normalize to remove the time
				daily_avg["date"] = pd.to_datetime(daily_avg["date"]).dt.normalize()

				# 🧼 Clean and prepare the data
				daily_avg["date"] = pd.to_datetime(daily_avg["date"]).dt.normalize()
				daily_avg = daily_avg[
					daily_avg["aqi"].notna() & (daily_avg["aqi"] != 0) &
					daily_avg["PM2_5"].notna() & (daily_avg["PM2_5"] != 0)
				]
				daily_avg["weekday"] = daily_avg["date"].dt.strftime("%a")  # e.g., Mon, Tue

				# 📊 Melt to long format for dual bar chart
				chart_df = daily_avg[["weekday", "PM2_5", "aqi"]].melt(id_vars="weekday", var_name="Metric", value_name="Value")

				# Define custom colors
				custom_color = alt.Scale(
					domain=["PM2_5", "aqi"],
					range=["#1f77b4", "#87CEFA"]  # dark blue for PM2.5, light blue for AQI
				)

				# Bar chart
				bar_chart = (
					alt.Chart(chart_df)
					.mark_bar()
					.encode(
						x=alt.X(
							"weekday:N", 
							title="Day of Week", 
							sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
						),
						y=alt.Y("Value:Q", title="Value (µg/m³ or AQI)"),
						color=alt.Color("Metric:N", scale=custom_color),
						tooltip=["Metric", "Value"]
					)
					.properties(
						height=240,
						width=700,
					)
					.configure_view(
						fill='white'  # sets chart background
					)
				)

				st.altair_chart(bar_chart,use_container_width=True)

		# Compute daily averages per station
		df_today_avg = df_today.groupby("station")[["aqi", "pm25"]].mean().reset_index()
		df_today_avg.rename(columns={"pm25": "pm25"}, inplace=True)

		
		# RIGHT COLUMN: Top 5 stations
		with right_col:

			with st.container(key="right_box"):
				st.markdown("""
				<div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">
					Highest AQI Today
				</div>
				<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
					Top 5 region with the highest PM2.5 and AQI for today
				</div>
				""", unsafe_allow_html=True)
				df_today_avg["color"] = df_today_avg["aqi"].apply(get_rgba_color)

				# Top 5
				top5_today = df_today_avg.sort_values("aqi", ascending=False).head(5)
				top5_today = top5_today.rename(columns={"pm25": "pm25"})

				for i, row in enumerate(top5_today.itertuples(index=False), start=1):
					station = row.station
					aqi = row.aqi
					pm25 = row.pm25
					color = row.color

					st.markdown(f"""
					<div style="
						background-color: #fdfdfd;
						border-left: 5px solid {color};
						padding: 14px 14px;
						border-radius: 8px;
						margin-bottom: 10px;
						box-shadow: 0 1px 2px rgba(0,0,0,0.08);
					">
						<div style="font-size: 14px; font-weight: bold;">
							#{i} {station}
						</div>
						<div style="font-size: 12px;">
							AQI: <b>{int(aqi)}</b> | PM2.5: <b>{pm25:.1f} µg/m³</b>
						</div>
					</div>
					""", unsafe_allow_html=True)

			

			with st.container(key="right_box_low"):
				# Bottom 5
				st.markdown("""
				<div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">
					Lowest AQI Today
				</div>
				<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
					Top 5 region with the lowest PM2.5 and AQI for today
				</div>
				""", unsafe_allow_html=True)
				low5_today = df_today_avg.sort_values("aqi", ascending=True).head(5)
				low5_today = low5_today.rename(columns={"pm25": "pm25"})

				for i, row in enumerate(low5_today.itertuples(index=False), start=1):
					station = row.station
					aqi = row.aqi
					pm25 = row.pm25
					color = row.color

					st.markdown(f"""
					<div style="
						background-color: #fdfdfd;
						border-left: 5px solid {color};
						padding: 14px 14px;
						border-radius: 8px;
						margin-bottom: 10px;
						box-shadow: 0 1px 2px rgba(0,0,0,0.08);
					">
						<div style="font-size: 14px; font-weight: bold;">
							#{i} {station}
						</div>
						<div style="font-size: 12px;">
							AQI: <b>{int(aqi)}</b> | PM2.5: <b>{pm25:.1f} µg/m³</b>
						</div>
					</div>
					""", unsafe_allow_html=True)					

#  PAGE 2: FILTER & DOWNLOAD
elif page == "Download Data":
	st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								Raw Data
							</div>
						""", unsafe_allow_html=True)
	css = """
		.st-key-selector_box {
			background-color: white;
			padding: 20px;
			border-radius: 10px;
			margin-bottom: 20px;
		}
		"""
	st.html(f"<style>{css}</style>")

	# 🔘 Selectors with custom container
	with st.container(key="selector_box"):
		st.markdown(f"""
							<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">
								Download Air Quality Data
							</div>
							<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
								The dataset utilized on this website is available for download. It is provided in a tabular format to facilitate analysis and integration into your projects.
							</div>
						""", unsafe_allow_html=True)

		@st.cache_data(show_spinner=True)
		def load_filtered_data(source_id=None, stations=None, date_range=None):
			conn = get_connection()

			# Start base query
			query = """
				SELECT station, sourceid, time, aqi, pm25, latitude, longitude
				FROM aqi
				WHERE 1=1
			"""

			params = []

			# Apply filters
			if source_id:
				query += " AND sourceid = %s"
				params.append(source_id)

			if stations and len(stations) > 0:
				placeholders = ", ".join(["%s"] * len(stations))
				query += f" AND station IN ({placeholders})"
				params.extend(stations)

			if date_range and len(date_range) == 2:
				start, end = date_range
				query += " AND time BETWEEN %s AND %s"
				params.extend([start, end])

			query += " ORDER BY time ASC"

			df = pd.read_sql(query, conn, params=params)
			conn.close()
			return df


		# -------------------------------
		# 1️⃣ Filter Form
		# -------------------------------
		conn = get_connection()
		source_ids = pd.read_sql("SELECT DISTINCT sourceid FROM aqi", conn)["sourceid"].sort_values().tolist()
		conn.close()

		with st.form("filter_form"):
			source_id = st.selectbox("Source ID", options=source_ids)

			# Load stations for selected source dynamically
			conn = get_connection()
			station_query = "SELECT DISTINCT station FROM aqi WHERE sourceid = %s ORDER BY station"
			station_options = pd.read_sql(station_query, conn, params=[source_id])["station"].tolist()
			conn.close()

			station_filter = st.multiselect("Station", options=station_options)
			date_range = st.date_input("Date range", [])
			submit = st.form_submit_button("Apply Filters")

		# -------------------------------
		# 2️⃣ Load data (with filters)
		# -------------------------------
		if submit:
			df_filtered = load_filtered_data(source_id, station_filter, date_range)
			st.write(f"Filtered rows: {len(df_filtered)}")

			# Pagination
			page_size = 1000
			max_page = (len(df_filtered) - 1) // page_size + 1

			if "page_num" not in st.session_state:
				st.session_state.page_num = 1

			col_prev, col_info, col_next = st.columns([1, 2, 1])
			with col_prev:
				if st.button("Prev") and st.session_state.page_num > 1:
					st.session_state.page_num -= 1
			with col_info:
				st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page_num} of {max_page}</div>", unsafe_allow_html=True)
			with col_next:
				if st.button("Next") and st.session_state.page_num < max_page:
					st.session_state.page_num += 1

			start = (st.session_state.page_num - 1) * page_size
			end = start + page_size
			st.dataframe(df_filtered.iloc[start:end])

			# Download CSV
			csv = df_filtered.to_csv(index=False).encode("utf-8")
			st.download_button("Download as CSV", data=csv, file_name="air_quality_filtered.csv", mime="text/csv")

#  PAGE 3: MODEL PREDICTION
elif page == "AOD Derived PM2.5 Heatmap":
	st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								AOD Derived PM2.5 Heatmap Over Jakarta
							</div>
						""", unsafe_allow_html=True)

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
				PM2.5 Prediction Using Aerosol Optical Depth
			</div>

			<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
				This heatmap visualizes the predicted PM2.5 concentrations, which are a key indicator of ambient air quality and potential health risks. Satellite-derived Aerosol Optical Depth (AOD) has been extensively studied as a proxy for surface-level PM2.5. For instance, Paciorek et al. (2008) identified statistically significant spatiotemporal associations between AOD retrievals and ground-level PM2.5 in the eastern United States. In our current setup, we utilize a traditional machine learning algorithms, <b>XGBoost</b>, <b>Random Forest</b>, and <b>LightGBM</b>, with AOD, meteorological parameters, and land-use features as predictors. The model is retrained weekly using the latest observed PM2.5 data to support continuous validation and improvement.
			</div>

			<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
				The heatmap is generated from tabular spatial data that has been converted into <b>GeoDataFrames</b> using the <b>GeoPandas</b> library, with a spatial resolution of approximately 800 meters. Model performance is evaluated by comparing predicted and observed PM2.5 values from monitoring stations using the <b>Mean Squared Error (MSE)</b> metric.
			</div>
			""", unsafe_allow_html=True)
			
		st.markdown("<br>", unsafe_allow_html=True)

	css = """
		.st-key-selector_box {
			background-color: white;
			padding: 20px;
			border-radius: 10px;
			margin-bottom: 20px;
		}
		"""
	st.html(f"<style>{css}</style>")

	# 🔘 Selectors with custom container
	with st.container(key="selector_box"): 

		import os
		import streamlit as st
		import folium
		from streamlit_folium import st_folium
		from folium.plugins import MiniMap
		import rasterio
		import numpy as np
		from matplotlib import colors, cm
		import pandas as pd
		from branca.colormap import LinearColormap


		# --- Input folders ---
		tif_dir = "/app/tif_output/"
		models = ["xgb", "rf", "lgbm"]

		# --- Load daily station dataframe ---
		df_pm25_daily = pd.read_csv("/app/tif_output/daily_complete.csv", parse_dates=["date"])

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

		# --- Create Folium Map ---
		m = folium.Map(location=[-6.3, 106.8], zoom_start=10, tiles="cartodbpositron")

		
		vmin, vmax = 0, 80
		cmap = cm.turbo
		norm = colors.Normalize(vmin=vmin, vmax=vmax)

		# --- Load raster ---
		if selected_date == "All Dates":
			raster_stack = []
			for tif_file in tifs:
				with rasterio.open(os.path.join(tif_dir, tif_file)) as src_temp:
					img_temp = src_temp.read(1).astype(float)
					if src_temp.nodata is not None:
						img_temp = np.where(img_temp == src_temp.nodata, np.nan, img_temp)
					raster_stack.append(img_temp)
					bounds = src_temp.bounds  # store bounds from last file
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
			stations_today = (
				df_pm25_daily
				.groupby(["latitude", "longitude"], as_index=False)
				.agg({f"pm25_{selected_model}": "mean"})
			)
		else:
			stations_today = df_pm25_daily[df_pm25_daily["date"] == pd.to_datetime(selected_date)]

		# --- Overlay stations and compute errors ---
		errors = []
		for _, row in stations_today.iterrows():
			pm25_station = row[f"pm25_{selected_model}"]

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

		minimap = MiniMap(tile_layer=carto_tiles, toggle_display=True, position="bottomright")
		m.add_child(minimap)

		folium.LayerControl().add_to(m)

		st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								Station vs Estimated PM2.5 Map
							</div>
						""", unsafe_allow_html=True)

		# --- Show map ---
		st_map = st_folium(m, use_container_width=True, height=500)

		st.markdown("<br>", unsafe_allow_html=True)

		# --- Show error table and scatter ---
		if selected_date == "All Dates":
			st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								Station vs Estimated Errors (All Dates)
							</div>
						""", unsafe_allow_html=True)

			# Pick the raster column based on the selected model
			raster_col = f"pm25_{selected_model}"

			# Load compiled daily dataset
			df_all = pd.read_csv("/app/tif_output/daily_complete.csv")

			# Rename station value column if needed internally
			df_all = df_all.rename(columns={"PM2.5": "station_val"})

			# Select only the columns you want
			df_errors = df_all[["station", "station_val", raster_col]].copy()

			# Compute error if needed
			df_errors["error"] = df_errors["station_val"] - df_errors[raster_col]

		else:
			st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								Station vs Raster Errors ({selected_date})
							</div>
						""", unsafe_allow_html=True)

			# Convert the errors list to DataFrame if not already
			df_errors = pd.DataFrame(errors)

			# Pick the raster column based on the selected model
			raster_col = f"pm25_{selected_model}"

			# Compute error if needed
			df_errors["error"] = df_errors["station_val"] - df_errors[raster_col]

		# --- Display the table with nice column headers ---
		display_columns = {
			"station": "Station",
			"station_val": "Station PM2.5",
			raster_col: f"Estimated PM2.5 ({selected_model})",
			"error": "Error"
		}

		st.dataframe(df_errors.rename(columns=display_columns))

		st.markdown("<br>", unsafe_allow_html=True)

		import pandas as pd
		import numpy as np
		import altair as alt
		from sklearn.metrics import r2_score, mean_absolute_error

		if not df_errors.empty:
			st.markdown(f"""
							<div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
								Scatter Station vs Estimated PM2.5
							</div>
						""", unsafe_allow_html=True)

			df_errors = df_errors.dropna()

			# Use display names for clarity
			obs_col = "station_val"
			pred_col = f"pm25_{selected_model}"
			obs_display = "Station PM2.5 (µg/m³)"
			pred_display = f"Estimated PM2.5 ({selected_model.upper()})"

			obs = df_errors[obs_col]
			pred = df_errors[pred_col]

			# --- Calculate metrics ---
			R2 = r2_score(obs, pred)
			MAE = mean_absolute_error(obs, pred)
			RMSE = np.square(MAE)
			Bias = np.mean(pred - obs)
			RPE = np.mean(np.abs(pred - obs) / obs) * 100

			# --- Scatter plot ---
			scatter = (
				alt.Chart(df_errors)
				.mark_circle(size=80, opacity=0.7)
				.encode(
					x=alt.X(obs_col, title=obs_display),
					y=alt.Y(pred_col, title=pred_display),
					color=alt.Color("error", scale=alt.Scale(scheme="redyellowgreen")) 
						if "error" in df_errors.columns else alt.value("steelblue"),
					tooltip=[alt.Tooltip(c, title=display_columns.get(c, c)) for c in df_errors.columns]
				)
				.properties(width=700, height=400)
				.interactive()
			)

			# --- 1:1 reference line ---
			ref_line = (
				alt.Chart(pd.DataFrame({"x": [vmin, vmax], "y": [vmin, vmax]}))
				.mark_line(color="black", strokeDash=[4, 4])
				.encode(x="x", y="y")
			)

			# --- Metrics overlay at bottom-right ---
			metrics_text = pd.DataFrame({
				"metrics": [f"R²: {R2:.2f}", f"MAE: {MAE:.2f}", f"RMSE: {RMSE:.2f}",
							f"Bias: {Bias:.2f}", f"RPE: {RPE:.2f}%"],
				"x": [vmax]*5,  # all the way to the right
				"y": [(vmin+1) + i*(vmax-vmin)*0.03 for i in range(5)]  # small vertical offsets from bottom
			})

			metrics = (
				alt.Chart(metrics_text)
				.mark_text(align='right', fontSize=10, fontWeight='bold', color='black')
				.encode(
					x='x',
					y='y',
					text='metrics'
				)
				)


			st.altair_chart(scatter + ref_line + metrics, use_container_width=True)

#  PAGE 4: ABOUT
elif page == "About":
	csstab = """
		.st-key-about_site {
			background-color: white;
			padding: 20px;
			border-radius: 10px;
			margin-bottom: 20px;
			font-size: 14px;
			line-height: 1.6;
		}
	"""
	st.html(f"<style>{csstab}</style>")

	with st.container(key="about_site"):
		st.markdown("""
			<div style="font-size:18px; font-weight:600; margin-bottom:15px;">
				About This Project
			</div>

			<p>
			This platform compiles real-time and historical air quality data for Jakarta from four independent API sources. 
			The idea is so that users can explore and download the complete dataset for their own analysis or projects.  
			Beyond station measurements, the platform predicts PM2.5 concentrations for any latitude–longitude coordinate in Jakarta, 
			providing estimates in areas without direct monitoring coverage.
			</p>

			<div style="font-size:16px; font-weight:500; margin-top:20px; margin-bottom:10px;">
				Technical Overview
			</div>
			<ul>
				<li>Compile and process PM2.5 data from different APIs using <b>Python</b>.</li>
				<li><b>PostgreSQL</b> database for efficient data storage and retrieval.</li>
				<li>Containerized with <b>Docker</b> and deployed on an <b>Ubuntu</b> server.</li>
				<li>Built using <b>Streamlit</b> with integrated UI components and custom assets.</li>
				<li>Run machine learning models locally then upload it to the database.</li>
				<li>Served through <b>NGINX</b> for performance and reliability.</li>
			</ul>

			<div style="font-size:16px; font-weight:500; margin-top:20px; margin-bottom:10px;">
				References
			</div>
			<ul>
				<li>Xue, T., Zheng, Y., Geng, G., Zheng, B., Jiang, X., Zhang, Q., & He, K. (Year). "Fusing Observational, Satellite Remote Sensing and Air Quality Model Simulated Data to Estimate Spatiotemporal Variations of PM2.5 Exposure in China."</li>
				<li>Paciorek, C. J., et al. (2008). "Spatiotemporal associations between satellite-derived aerosol optical depth and PM2.5 in the eastern United States."</li>
				<li><a href="https://www.iqair.com/us/indonesia/jakarta" target="_blank">IQAir Jakarta</a></li>
				<li><a href="https://rendahemisi.jakarta.go.id/ispu" target="_blank">Jakarta Rendah Emisi</a></li>
				<li><a href="https://aqicn.org/network/menlhk/id/" target="_blank">Kementerian Lingkungan Hidup dan Kehutanan (KLHK)</a></li>
				<li><a href="https://id.usembassy.gov/u-s-embassy-jakarta-air-quality-monitor/" target="_blank">Udara Jakarta</a></li>
				<li><a href="https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5" target="_blank">ERA5 (ECMWF Reanalysis v5)</a></li>
				<li><a href="https://developers.google.com/earth-engine/datasets/tags/weather" target="_blank">Google Earth Engine</a></li>
			</ul>

			<div style="margin-top:20px; font-size:14px;">
				<p>This project is currently in an early stage of development and will improve over time.</p>
				<p>Connect with me on <a href="https://www.linkedin.com/in/yourprofile/" target="_blank">LinkedIn</a>.</p>
			</div>
			""", unsafe_allow_html=True)
