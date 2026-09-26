"""
WeatherBlend AI - Backend Runner Script
Launches FastAPI Uvicorn Server
"""
import uvicorn

if __name__ == "__main__":
    print("Starting WeatherBlend AI Backend API Server on http://127.0.0.1:8000...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
