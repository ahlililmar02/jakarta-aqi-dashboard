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
import streamlit as st
from streamlit_option_menu import option_menu

# Import pages
from modules import air_quality_monitor, download_data, aod_pm25_heatmap, about

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


with open("static/style.css") as f:
	st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#  PAGE 1: DASHBOARD
if page == "Air Quality Monitor":
    air_quality_monitor.show()

#  PAGE 2: FILTER & DOWNLOAD
elif page == "Download Data":
	download_data.show()

#  PAGE 3: MODEL PREDICTION
elif page == "AOD Derived PM2.5 Heatmap":
    aod_pm25_heatmap.show()

#  PAGE 4: ABOUT
elif page == "About":
    about.show()
