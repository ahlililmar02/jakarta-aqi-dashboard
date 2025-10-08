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

def show():
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
				<li>Store and manage processed data efficiently in a <b>PostgreSQL</b> database.</li>
				<li>Containerized with <b>Docker</b> and deployed on an <b>Ubuntu</b> server.</li>
				<li>Hosted and orchestrated on <b>Google Cloud Platform (GCP)</b> for scalability and reliability.</li>
				<li>Implemented machine learning models</b> for PM2.5 spatiotemporal prediction.</li>
				<li>Utilized <b>Gemini AI for data analysis and anomaly detection.</li>				
				<li>Visualized results through an interactive <b>Streamlit</b> dashboard with custom UI components.</li>
				<li>Served securely via <b>NGINX</b> for optimized performance and uptime.</li>
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
				<p>Connect with me on <a href="https://www.linkedin.com/in/ahlil-batuparan-850b6b243/" target="_blank">LinkedIn</a>.</p>
				<p>View my project source code on <a href="https://www.linkedin.com/in/ahlil-batuparan-850b6b243/" target="_blank">GitHub</a>.</p>
			</div>
			""", unsafe_allow_html=True)
