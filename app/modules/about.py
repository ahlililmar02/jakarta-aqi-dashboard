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

			<div style="margin-top:20px; font-size:14px;">
				<p>This project is currently in an early stage of development and will improve over time.</p>
				<p>Connect with me on <a href="https://www.linkedin.com/in/ahlil-batuparan-850b6b243/" target="_blank">LinkedIn</a>.</p>
				<p>View my project on <a href="https://github.com/ahlililmar02" target="_blank">GitHub</a>.</p>
			</div>

			<div style="font-size:16px; font-weight:500; margin-top:20px; margin-bottom:10px;">
				References
			</div>
			<ul>
				<li>Cooper, D. C., & Alley, F. C. (1994). "Air Pollution Control: A Design Approach."</li>
				<li>Grzybowski, P. T., Markowicz, K. M., & Musiał, J. P. (2023). "Estimations of the ground-level NO₂ concentrations based on the Sentinel-5P NO₂ tropospheric column number density product." <a href="https://doi.org/10.3390/rs15020378" target="_blank">Link</a></li>
				<li>Paciorek, C. J., et al. (2008). "Spatiotemporal associations between satellite-derived aerosol optical depth and PM2.5 in the eastern United States."</li>
				<li>Rahman, R. N., Farda, N. M., & Sopaheluwakan, A. (2025). "Estimation of PM₂.₅ concentration in DKI Jakarta from Sentinel-5P imagery by considering meteorological factors using Random Forest approach." <a href="https://doi.org/10.1051/bioconf/202516703003" target="_blank">Link</a></li>
				<li>Shetty, S., Schneider, P., Stebel, K., Hamer, P. D., Kylling, A., Berntsen, T. K., & Koren, T. (2024). "Estimating surface NO₂ concentrations over Europe using Sentinel-5P TROPOMI observations and machine learning." <a href="https://doi.org/10.1016/j.rse.2024.114321" target="_blank">Link</a></li>
				<li>Syuhada, G., Akbar, A., Hardiawan, D., Pun, V., Darmawan, A., Heryati, S. H. A., Siregar, A. Y. M., Kusuma, R. R., Driejana, R., Ingole, V., Kass, D., & Mehta, S. (2023). "Impacts of air pollution on health and cost of illness in Jakarta, Indonesia." <a href="https://doi.org/10.3390/ijerph20042916" target="_blank">Link</a></li>
				<li>Thongthammachart, T., Araki, S., Shimadera, H., Matsuo, T., & Kondo, A. (2022). "Incorporating Light Gradient Boosting Machine to land use regression model for estimating NO₂ and PM₂.₅ levels in Kansai region, Japan." <a href="https://doi.org/10.1016/j.envsoft.2022.105447" target="_blank">Link</a></li>
				<li>Xue, T., Zheng, Y., Geng, G., Zheng, B., Jiang, X., Zhang, Q., & He, K. (2017). "Fusing observational, satellite remote sensing and air quality model simulated data to estimate spatiotemporal variations of PM₂.₅ exposure in China." <a href="https://doi.org/10.3390/rs9030221" target="_blank">Link</a></li>
				<li>Zamani Joharestani, M., Cao, C., Ni, X., Bashir, B., & Talebiesfandarani, S. (2019). "PM₂.₅ prediction based on Random Forest, XGBoost, and deep learning using multisource remote sensing data." <a href="https://doi.org/10.3390/atmos10070373" target="_blank">Link</a></li>
			</ul>
			<div style="font-size:16px; font-weight:500; margin-top:20px; margin-bottom:10px;">
				Dataset
			</div>

			<ul>	
				<li><a href="https://www.iqair.com/us/indonesia/jakarta" target="_blank">IQAir Jakarta</a></li>
				<li><a href="https://rendahemisi.jakarta.go.id/ispu" target="_blank">Jakarta Rendah Emisi</a></li>
				<li><a href="https://aqicn.org/network/menlhk/id/" target="_blank">Kementerian Lingkungan Hidup dan Kehutanan (KLHK)</a></li>
				<li><a href="https://id.usembassy.gov/u-s-embassy-jakarta-air-quality-monitor/" target="_blank">Udara Jakarta</a></li>
				<li><a href="https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5" target="_blank">ERA5 (ECMWF Reanalysis v5)</a></li>
				<li><a href="https://developers.google.com/earth-engine/datasets/tags/weather" target="_blank">Google Earth Engine</a></li>
			</ul>
						
			""", unsafe_allow_html=True)
