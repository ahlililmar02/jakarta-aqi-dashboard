import os
import psycopg2
import pandas as pd
from datetime import datetime
import streamlit as st

def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASS")
    )