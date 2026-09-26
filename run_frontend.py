"""
WeatherBlend AI - Frontend Runner Script
Launches Streamlit Dashboard App
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "app.py")
    print(f"Launching Streamlit Frontend Application: {app_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
