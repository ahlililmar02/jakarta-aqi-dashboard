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
from datetime import datetime, timedelta
from google import genai
from google.genai import types

# Create client
client = genai.Client(vertexai=True, project="double-reef-468803-r9", location="us-central1")


@st.cache_data(ttl=1800) 
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
			AND pm25 < 500
		"""
	df_today = pd.read_sql(query, conn)
	conn.close()
	return df_today

def show():
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
	if st.button("Refresh Data"):
		st.cache_data.clear()
		st.rerun()
		
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

	with st.container(key="selector_box"):
		from datetime import datetime, timedelta

		sourceid_list = df_today["sourceid"].unique()

		# Set default for source ID (e.g., first one or a specific value)
		default_source = sourceid_list[-1]  # or 'SOME_SOURCE_ID' if you know the ID
		selected_source = st.selectbox("Select Source ID", sourceid_list, index=list(sourceid_list).index(default_source))
		
		now = datetime.now()

		# Filter df_today to rows within the last 3 hours
		recent_df = df_today[df_today['time'] >= now - timedelta(hours=4)]

		# Filter by selected source
		recent_source_df = recent_df[recent_df["sourceid"] == selected_source]

		# Get unique stations with recent data
		stations_in_source = recent_source_df["station"].unique()

		if len(stations_in_source) == 0:
			st.warning("No stations have data within the last 4 hours for this source.")
			selected_station = None
		else:
			# Set default station (first one)
			default_station = stations_in_source[-1]
			selected_station = st.selectbox(
				"Select Station",
				stations_in_source,
				index=list(stations_in_source).index(default_station)
			)

			# Get the station's data within last 3 hours
			station_df = recent_source_df[recent_source_df["station"] == selected_station].sort_values("time")

			if not station_df.empty:
				latest_row = station_df.iloc[-1]  # latest row within last 3 hours
			else:
				latest_row = None
		center = [latest_row["latitude"], latest_row["longitude"]]

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

	#minimap = MiniMap(tile_layer=carto_tiles, toggle_display=True, position="bottomright")
	#m.add_child(minimap)

	from folium.features import DivIcon
	# Filter by sourceid first
	filtered_df = df_latest[df_latest["sourceid"] == selected_source]

	# Then filter to rows within the last 3 hours
	filtered_df = filtered_df[filtered_df['time'] >= now - timedelta(hours=4)]

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
					border:{border};
					text-shadow:
						0 0 1px black,
						0 0 1px black;
					'>
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
	.st-key-metric_box, .st-key-map {
		background-color: white;
		padding: 20px;
		border-radius: 10px;
		margin-bottom: 20px;
	}
	"""
	st.html(f"<style>{css2}</style>")

	#map_col, space_col, metric_col = st.columns([3.5, 0.01, 1.5])

	st.markdown('<div style="position: relative;">', unsafe_allow_html=True)

	with st.container(key="map"):
		map_output = st_folium(
			m,
			height=500,
			use_container_width=True,
			returned_objects=["last_object_clicked"]
		)

		# --- Handle map click ---
		if map_output and map_output.get("last_object_clicked"):
			lat_click = map_output["last_object_clicked"]["lat"]
			lon_click = map_output["last_object_clicked"]["lng"]

			# Calculate squared distance to find nearest station
			df_today["distance"] = (
				(df_today["latitude"] - lat_click)**2 + (df_today["longitude"] - lon_click)**2
			)
			nearest_station = df_today.sort_values("distance").iloc[0]["station"]
			st.session_state.selected_station = nearest_station
			selected_station = nearest_station
			
			station_df = df_today[df_today["station"] == selected_station].sort_values("time")
			# Filter rows within the last 3 hours
			recent_rows = station_df[station_df['time'] >= now - timedelta(hours=3)]

			if not recent_rows.empty:
				latest_row = recent_rows.iloc[-1]  # latest row within last 3 hours
			else:
				latest_row = None  



		def get_aqi_category(aqi_value):
				"""Return AQI category label based on US EPA standard."""
				if aqi_value <= 50:
					return "Good"
				elif aqi_value <= 100:
					return "Moderate"
				elif aqi_value <= 150:
					return "Unhealthy for Sensitive Group"
				elif aqi_value <= 200:
					return "Unhealthy"
				elif aqi_value <= 300:
					return "Very Unhealthy"
				else:
					return "Hazardous"
		color = get_rgba_color(latest_row["aqi"])
			
		category = get_aqi_category(latest_row["aqi"])
		
		# 📘 Legend
		legend_html = """
		<div style="font-family: 'Inter', sans-serif;">
			<h3 style="font-size: 18px; margin-bottom: 10px;">AQI Categories (US EPA Standard)</h3>
			<div style="display: flex; flex-wrap: wrap; gap: 15px; font-size: 13px;">
				<div style="display: flex; align-items: center; gap: 4px;">
					<div style="background-color: rgba(0, 228, 0, 0.7); width: 14px; height: 14px; border-radius: 3px; border: 1px solid #000;"></div> Good (0–50)
				</div>
				<div style="display: flex; align-items: center; gap: 4px;">
					<div style="background-color: rgba(255, 255, 0, 0.7); width: 14px; height: 14px; border-radius: 3px; border: 1px solid #000;"></div> Moderate (51–100)
				</div>
				<div style="display: flex; align-items: center; gap: 4px;">
					<div style="background-color: rgba(255, 126, 0, 0.7); width: 14px; height: 14px; border-radius: 3px; border: 1px solid #000;"></div> Unhealthy for SG (101–150)
				</div>
				<div style="display: flex; align-items: center; gap: 4px;">
					<div style="background-color: rgba(255, 0, 0, 0.7); width: 14px; height: 14px; border-radius: 3px; border: 1px solid #000;"></div> Unhealthy (151–200)
				</div>
				<div style="display: flex; align-items: center; gap: 4px;">
					<div style="background-color: rgba(143, 63, 151, 0.7); width: 14px; height: 14px; border-radius: 3px; border: 1px solid #000;"></div> Very Unhealthy (201–300)
				</div>
				<div style="display: flex; align-items: center; gap: 4px;">
					<div style="background-color: rgba(126, 0, 35, 0.7); width: 14px; height: 14px; border-radius: 3px; border: 1px solid #000;"></div> Hazardous (301+)
				</div>
			</div>
		</div>
		"""
		st.markdown(legend_html, unsafe_allow_html=True)
		st.markdown("<br>", unsafe_allow_html=True)



	def get_category_icon(aqi):
		if aqi <= 50:
			return "😊"
		elif aqi <= 100:
			return "🙂"
		elif aqi <= 150:
			return "😷"
		elif aqi <= 200:
			return "🤒"
		elif aqi <= 300:
			return "😫"
		else:
			return "☠️"

	icon = get_category_icon(latest_row["aqi"])

	overlay_html = f"""
		<style>
		.overlay-aqi {{
			position: absolute;
			bottom: 500px;
			right: 40px;
			background-color: {color};
			border-radius: 12px;
			padding: 14px 18px;
			color: #111;
			font-family: 'Inter', sans-serif;
			box-shadow: 0 4px 10px rgba(0,0,0,0.25);
			width: 280px;
		}}
		.aqi-row {{
			display: flex;
			justify-content: space-between;
			align-items: center;
		}}
		.aqi-value {{
			font-size: 28px;
			font-weight: 700;
			line-height: 1;
		}}
		.aqi-sub {{
			font-size: 11px;
			font-weight: 400;
			opacity: 0.8;
		}}
		.aqi-station {{
			font-weight: 600;
			font-size: 11px;
		}}
		.aqi-category {{
			font-size: 14px;
			font-weight: 600;
			margin-top: 2px;
		}}
		.aqi-icon {{
			font-size: 36px;
		}}
		.aqi-footer {{
			margin-top: 10px;
			font-size: 13px;
			display: flex;
			justify-content: space-between;
			border-top: 1px solid rgba(0,0,0,0.1);
			padding-top: 6px;
		}}
		
		/* MOBILE STYLES */
		@media (max-width: 400px) {{
			.overlay-aqi {{
				width: 60vw;               /* fill most of screen width */
				padding: 10px 12px;
				bottom: 670px;               /* a bit closer to bottom */
				right: 30px;
			}}
			.aqi-value {{ font-size: 22px; }}
			.aqi-sub {{ font-size: 10px; }}
			.aqi-station {{ font-size: 10px; }}
			.aqi-icon {{ font-size: 28px; margin-top: 6px; }}
			.aqi-category {{ font-size: 12px; }}
			.aqi-footer {{ font-size: 11px; }}
		}}
		</style>

		<div class="overlay-aqi">
			<div class="aqi-row">
				<div>
				<div class="aqi-value">{int(latest_row["aqi"])}</div>
				<!-- Station sits inline next to "AQI" -->
				<div class="aqi-sub">AQI at <span class="aqi-station">{latest_row["station"]}</span></div>
				</div>
				<div class="aqi-icon">{icon}</div>
			</div>
			<div class="aqi-category">{category}</div>
			<div class="aqi-footer">
				<div>{latest_row["time"].strftime('%H:%M')}</div>
				<div>{latest_row["pm25"]:.1f} µg/m³</div>
			</div>
		</div>
		"""

	st.markdown(overlay_html, unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)
	
	
	st.html("""
		<style>
		.st-key-current_box,.st-key-right_box,.st-key-right_box_low, .st-key-time_series, .st-key-bar_chart {
			background-color: white;
			padding: 16px 16px;
			border-radius: 8px;
			margin-bottom: 7px;
		}
		</style>
		""")

	left_col, middle_col, right_col = st.columns([2.5, 0.01, 1.8])
	st.markdown("""
		<style>
		.full-height {
			display: flex;
			flex-direction: column;
			justify-content: space-between;
			height: 100%;
		}
		</style>
		""", unsafe_allow_html=True)
	# 8. Split into 3 columns: left = metrics + chart, middle = space, right = top 5 AQI
	with st.container():
		with left_col:
			st.markdown('<div class="full-height">', unsafe_allow_html=True)

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

					# Prepare DataFrame
					metric_option = st.selectbox("", ["AQI", "PM2.5"],label_visibility="collapsed")

					# --- Prepare DataFrame ---
					df_plot = station_df.copy()
					df_plot["time"] = pd.to_datetime(df_plot["time"])
					df_plot = df_plot.sort_values("time")


					# Assign color for each bar using AQI value
					df_plot["color"] = df_plot["aqi"].apply(lambda x: get_rgba_color(x, 0.7))

					# --- Choose which metric to show ---
					y_column = "aqi" if metric_option == "AQI" else "pm25"
					y_label = "Air Quality Index (AQI)" if metric_option == "AQI" else "PM2.5 (µg/m³)"

					# --- Create Altair bar chart ---
					chart = (
						alt.Chart(df_plot)
						.mark_bar(size=20,
							cornerRadiusTopLeft=4,
							cornerRadiusTopRight=4
						)
						.encode(
							x=alt.X("time:T", title="Time", axis=alt.Axis(format="%H:%M")),
							y=alt.Y(f"{y_column}:Q", title=y_label),
							color=alt.Color("color:N", scale=None, legend=None),
							tooltip=["time", y_column, "aqi", "pm25"]
						)
						.properties(width=700, height=250)
						.configure_view(fill='white')
						.configure_axis(labelPadding=8, titlePadding=12)
						.properties(padding={"left": 10, "right": 15, "top": 10, "bottom": 10})	
					)

					st.altair_chart(chart, use_container_width=True)
					st.markdown("""
					<div style="font-size:16px; font-weight:500; margin-bottom:8px;">
						Analysis
					</div>
					<div style="font-size:10px; font-weight:300; margin-bottom:2px;">
						Generated by Gemini AI based on today's PM2.5 and AQI data.
					</div>
					""", unsafe_allow_html=True)

					# Prepare context data
					daily_data = station_df[["time", "aqi", "pm25", "latitude", "longitude"]].tail(24)

					# Detect spikes today
					threshold = daily_data["pm25"].mean() + 1.5 * daily_data["pm25"].std()
					spikes = daily_data[daily_data["pm25"] > threshold]
					spike_info = ", ".join(spikes["time"].dt.strftime("%H:%M")) if not spikes.empty else "No spikes"

					# Build simple prompt
					prompt = f"""
					Summarize today's PM2.5 air quality for {selected_station}.
					
					Use **colored text** for air quality categories in HTML as follows:
						- <span style='color:#28a745;'>Good</span>
						- <span style='color:#ffc107;'>Moderate</span>
						- <span style='color:#fd7e14;'>Unhealthy for Sensitive Groups</span>
						- <span style='color:#dc3545;'>Unhealthy</span>
						- <span style='color:#6f42c1;'>Very Unhealthy</span>
						- <span style='color:#343a40;'>Hazardous</span>

					Today's PM2.5 data (hourly):
					{daily_data.to_csv(index=False)}

					Include:
					- Output in **bullet points**.
					- Today's hourly AQI and PM2.5 trend and spikes ({spike_info}).
					- Highlight categorization of air quality (Good, Moderate, Unhealthy, etc.). **colored text**.  
					- Bold critical information like spike times and unusually high readings.  
					- Optionally, include any relevant context based on location ({daily_data['latitude'].iloc[-1]} and longitude {daily_data['longitude'].iloc[-1]}) and hours, for example, industry activities considering the locations nearby the station.
					- Keep it concise (under 150 words) and **easy to read**..
					"""

					# --- Vertex AI call ---
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
							font-size:14px;     
							font-weight:300; 
							text-align:justify;">{response.text} </div>
						""", unsafe_allow_html=True)

					except Exception as e:
						st.error(f"Error calling Gemini AI: {e}")

			with st.container(key="bar_chart"):
				import altair as alt
				from datetime import datetime, timedelta

				today = datetime.now()

				def load_aqi_data(start_date, end_date):
					conn = get_connection()
					query = f"""
						SELECT station, sourceid, time, aqi, pm25, latitude, longitude
						FROM aqi
						WHERE time BETWEEN '{start_date}' AND '{end_date}'
						AND aqi IS NOT NULL
						AND aqi != 0
					"""
					df = pd.read_sql(query, conn)
					conn.close()
					return df

				# --- Time window selector ---
				#period_option = st.radio("Select Time Range",["Last 7 Days", "Last 30 Days"],horizontal=True)

				days_back = 7 #if period_option == "Last 7 Days" else 30
				start_date = today - timedelta(days=days_back)
				end_date = today

				start_str = start_date.strftime('%Y-%m-%d 00:00:00')
				end_str = end_date.strftime('%Y-%m-%d 23:59:59')

				# --- Load data and prepare ---
				df = load_aqi_data(start_str, end_str)
				df_station = df[df["station"] == selected_station].copy()
				df_station["date"] = pd.to_datetime(df_station["time"]).dt.date

				# --- Compute daily average ---
				daily_df = (
					df_station
					.groupby("date", as_index=False)
					[["aqi", "pm25"]]
					.mean()
				)
				daily_df["date"] = pd.to_datetime(daily_df["date"])
				daily_df = daily_df[daily_df["aqi"].notna() & (daily_df["aqi"] != 0)]

				# --- Apply AQI color ---
				daily_df["color"] = daily_df["aqi"].apply(lambda x: get_rgba_color(x, 0.8))

				st.markdown(f"""
					<div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">
						Last {days_back} days Trend — {selected_station}
					</div>

					<div style="font-size:14px; font-weight:300; margin-bottom:10px;">
						Daily PM2.5 and AQI time series for the last {days_back} days
					</div>
					""", unsafe_allow_html=True)


				# --- Metric selector ---
				metric_option = st.selectbox(
					"Select Metric",
					["AQI", "PM2.5"],
					key="metric_select",
					label_visibility="collapsed"  # hides the label entirely
				)

				# --- Base chart config ---
				base = alt.Chart(daily_df).encode(
					x=alt.X("date:T", title="Date")
				)

				if metric_option == "AQI":
					line = (
						base.mark_line(color="gray", opacity=0.3, strokeWidth=2)
						.encode(
							y=alt.Y("aqi:Q", title="Daily Average AQI"),
							tooltip=["date", "aqi"]
						)
					)

					points = (
						base.mark_circle(size=80)
						.encode(
							y="aqi:Q",
							color=alt.Color("color:N", scale=None, legend=None),
							tooltip=["date", "aqi"]
						)
					)

					# Add value labels (matching color)
					labels = (
						base.mark_text(dy=-12, fontSize=11)
						.encode(
							y="aqi:Q",
							text=alt.Text("aqi:Q", format=".0f"),
							color=alt.Color("color:N", scale=None, legend=None)
						)
					)

				else:  # PM2.5 selected
					line = (
						base.mark_line(color="gray", opacity=0.3, strokeWidth=2)
						.encode(
							y=alt.Y("pm25:Q", title="Daily Average PM2.5"),
							tooltip=["date", "pm25"]
						)
					)

					points = (
						base.mark_circle(size=80)
						.encode(
							y="pm25:Q",
							color=alt.Color("color:N", scale=None, legend=None),
							tooltip=["date", "pm25"]
						)
					)

					# Add value labels (blue)
					labels = (
						base.mark_text(dy=-12, fontSize=11)
						.encode(
							y="pm25:Q",
							text=alt.Text("pm25:Q", format=".0f"),
							color=alt.Color("color:N", scale=None, legend=None)
						)
					)


				# --- Combine chart ---
				line_chart = (
					(line + points + labels)
					.properties(
						height=250,
						width=700,
					)
					.configure_view(fill="white")
					.configure_axis(labelPadding=8, titlePadding=12)
					.configure_view(stroke=None)
					.properties(padding={"left": 10, "right": 15, "top": 10, "bottom": 10})
					)
				

				st.altair_chart(line_chart, use_container_width=True)

				st.markdown("""
						<div style="font-size:16px; font-weight:500; margin-bottom:8px;">
							Analysis
						</div>
						<div style="font-size:10px; font-weight:300; margin-bottom:2px;">
							Generated by Gemini AI based on the latest and weekly PM2.5 and AQI data.
						</div>
						""", unsafe_allow_html=True)

				# Prepare context data including location
				weekly_data = daily_df[["date", "aqi", "pm25"]].tail(7)

				# Detect high PM2.5 days in last 7 days
				weekly_threshold = weekly_data["pm25"].mean() + 1.5 * weekly_data["pm25"].std()
				high_days = weekly_data[weekly_data["pm25"] > weekly_threshold]

				# --- Example summary template for AI ---
				example_summary = """
					- **Today's PM2.5:** <span style='color:#ffc107;'>Moderate</span> Highest this week, likely due to weekend activities.  
					- **Daily Average Comparison:** Today's average (~62 µg/m³) is lower than yesterday decreasing about 12%.  
					- **Weekly Overview:** PM2.5 gradually decreased compared to earlier in the week.
					- **Notes:** No extreme <span style='color:#343a40;'>Hazardous</span> levels today. Recent high PM2.5 days were observed in the last week.
					"""

				# --- Build AI prompt ---
				prompt = f"""
					You are an environmental data analyst. Summarize the PM2.5 air quality data for {selected_station}
					Use **colored text** for air quality categories in HTML as follows:
					- <span style='color:#28a745;'>Good</span>
					- <span style='color:#ffc107;'>Moderate</span>
					- <span style='color:#fd7e14;'>Unhealthy for Sensitive Groups</span>
					- <span style='color:#dc3545;'>Unhealthy</span>
					- <span style='color:#6f42c1;'>Very Unhealthy</span>
					- <span style='color:#343a40;'>Hazardous</span>

					Weekly PM2.5 averages:
					{weekly_data.to_csv(index=False)}

					Example:
					{example_summary}

					Tasks:
					- Focus primarily on **AQI and PM2.5 trends**.  
					- Highlight periods of **high PM2.5 exposure** using **colored text**.  
					- Bold critical information like spikes and unusually high readings.  
					- Summarize weekly trends and compare today's trend to the previous days. 
					- Identify if today's PM2.5 levels are above or below the weekly average use {high_days}. 
					- Optionally, include any relevant context based on location ({daily_data['latitude'].iloc[-1]} and longitude {daily_data['longitude'].iloc[-1]}) and days.
					- Keep the summary concise (under 150 words) and **easy to read**.
					"""
					
				# --- Vertex AI call ---
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
							font-size:14px;     
							font-weight:300; 
							text-align:justify;">{response.text} </div>
						""", unsafe_allow_html=True)

				except Exception as e:
						st.error(f"Error calling Gemini AI: {e}")		

			# Compute daily averages per station
			df_today_avg = df_today.groupby("station")[["aqi", "pm25"]].mean().reset_index()
			st.markdown('</div>', unsafe_allow_html=True)
		
		# RIGHT COLUMN: Top 5 stations
		with right_col:
			st.markdown('<div class="full-height">', unsafe_allow_html=True)

			with st.container(key="current_box"):
				# --- Metric selector ---
				metric_option = st.selectbox(
					"Today's Overview",
					["AQI", "PM2.5"],
					index=0
				)

				# Determine column names based on metric
				if metric_option == "AQI":
					col_value = "aqi"
					label = "Air Quality Index (AQI)"
					threshold = 150
				else:
					col_value = "pm25"
					label = "PM2.5 (µg/m³)"
					threshold = 55

				# --- Compute latest, max, min and their times ---
				if not daily_data.empty:
					latest_val = daily_data[col_value].iloc[-1]
					latest_time = pd.to_datetime(daily_data["time"].iloc[-1]).strftime("%H:%M")

					max_val = daily_data[col_value].max()
					max_time = pd.to_datetime(daily_data["time"][daily_data[col_value].idxmax()]).strftime("%H:%M")

					min_val = daily_data[col_value].min()
					min_time = pd.to_datetime(daily_data["time"][daily_data[col_value].idxmin()]).strftime("%H:%M")
				else:
					latest_val = max_val = min_val = "-"
					latest_time = max_time = min_time = "-"

				color_latest = get_rgba_color(daily_data["aqi"].iloc[-1])
				color_max = get_rgba_color(daily_data["aqi"].max())
				color_min = get_rgba_color(daily_data["aqi"].min())

				st.markdown(f"""
				<div style="display:flex; flex-direction:column; gap:6px; margin-bottom:10px;">
					<!-- Latest -->
					<div style="background-color:{color_latest}; padding:10px; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display:flex; flex-direction:column;opacity: 0.9">
						<span style="font-size:10px; font-weight:400;">Latest</span>
						<div style="display:flex; justify-content:space-between; align-items:center;">
							<span style="font-size:14px; font-weight:600;">{latest_val}</span>
							<span style="font-size:14px; font-weight:300;">{latest_time}</span>
						</div>
					</div>
					<!-- Maximum -->
					<div style="background-color:{color_max}; padding:10px; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display:flex; flex-direction:column;opacity: 0.9">
						<span style="font-size:10px; font-weight:400;">Maximum</span>
						<div style="display:flex; justify-content:space-between; align-items:center;">
							<span style="font-size:14px; font-weight:600;">{max_val}</span>
							<span style="font-size:14px; font-weight:400;">{max_time}</span>
						</div>
					</div>
					<!-- Minimum -->
					<div style="background-color:{color_min}; padding:10px; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display:flex; flex-direction:column;opacity: 0.9">
						<span style="font-size:10px; font-weight:400;">Minimum</span>
						<div style="display:flex; justify-content:space-between; align-items:center;">
							<span style="font-size:14px; font-weight:600;">{min_val}</span>
							<span style="font-size:14px; font-weight:400;">{min_time}</span>
						</div>
					</div>
				</div>
				""", unsafe_allow_html=True)

				if latest_val > threshold:
					st.markdown(
						f"""
						<div style="
							background-color:#fff3cd;
							padding:8px 12px;
							border-radius:6px;
							box-shadow:0 1px 2px rgba(0,0,0,0.1);
							font-size:13px;
							text-align:justify;
							color:#856404;
						">
							⚠️ <b>{label}</b> reached <b>unhealthy</b> level ({latest_val:.1f}). 
							Please wear a mask near the area
						</div>
						""",
						unsafe_allow_html=True
					)


				# --- Example summary template for AI ---
				example_summary = """
					- **Overall Data:** The AQI values range from 134 to 182.
					- **Latest Data:** The latest reading at the selected station is <strong>higher than nearby stations</strong>, likely due to traffic.
					- **Suggestion:** Please wear mask considering the high concentration at that location.
					"""

				# --- Build AI prompt ---
				prompt = f"""
					You are an environmental data analyst. Summarize the PM2.5 air quality data for {selected_station} .

					Use **colored text** for air quality categories in HTML as follows:
					- <span style='color:#28a745;'>Good</span>
					- <span style='color:#ffc107;'>Moderate</span>
					- <span style='color:#fd7e14;'>Unhealthy for Sensitive Groups</span>
					- <span style='color:#dc3545;'>Unhealthy</span>
					- <span style='color:#6f42c1;'>Very Unhealthy</span>
					- <span style='color:#343a40;'>Hazardous</span>

					Today's PM2.5 data (hourly):
					{daily_data.to_csv(index=False)}

					an example of the summary format I want:
					{example_summary}

					Tasks:
					- Output in **bullet points**
					- Compare latest PM2.5 levels to nearby stations (latitude{daily_data['latitude'].iloc[-1]} and longitude {daily_data['longitude'].iloc[-1]}).  
					- Optionally, include any relevant context based on location (latitude {daily_data['latitude'].iloc[-1]} and longitude {daily_data['longitude'].iloc[-1]}).
					- Add a suggestion to the public
					- Keep the summary concise (under 110 words) and **easy to read**.
					"""
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
							font-size:12px;     
							font-weight:300; 
							text-align:justify;">{response.text} </div>
						""", unsafe_allow_html=True)

				except Exception as e:
						st.error(f"Error calling Gemini AI: {e}")


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
						<div style="font-size: 12px; font-weight: bold;">
							{i}. {station}
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
						<div style="font-size: 12px; font-weight: bold;">
							{i}. {station}
						</div>
						<div style="font-size: 12px;">
							AQI: <b>{int(aqi)}</b> | PM2.5: <b>{pm25:.1f} µg/m³</b>
						</div>
					</div>
					""", unsafe_allow_html=True)	
			
			st.markdown('</div>', unsafe_allow_html=True)	