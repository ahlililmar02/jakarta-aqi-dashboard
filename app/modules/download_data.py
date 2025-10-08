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

def show():
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
		source_ids = pd.read_sql(
			"SELECT DISTINCT sourceid FROM aqi WHERE sourceid != %s ORDER BY sourceid",
			conn,
			params=["JakartaRendahEmisi"]
		)["sourceid"].tolist()
		conn.close()

		with st.form("filter_form"):
			source_id = st.selectbox("Source ID", options=source_ids)

			# Load stations for selected source dynamically
			conn = get_connection()
			station_query = "SELECT DISTINCT station FROM aqi WHERE sourceid = %s ORDER BY station"
			station_options = pd.read_sql(station_query, conn, params=[source_id])["station"].tolist()
			conn.close()

			all_option = "All Stations"
			options = [all_option] + station_options

			station_filter = st.multiselect("Station (required)", options=options)

			# If "All Stations" is selected, override with all stations
			if all_option in station_filter:
				station_filter = station_options
				
			date_range = st.date_input("Date range (required)", [])

			st.warning("⚠️ Loading the whole dataset can be slow. Please consider the amount of data before proceeding.")

			submit = st.form_submit_button("Apply Filters")

		# Validate required fields after form submission
		if submit:
			if not station_filter:
				st.error("❌ Please select at least one station.")
			elif not date_range or len(date_range) != 2:
				st.error("❌ Please select a start and end date for the date range.")
			else:
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
