"""
Legacy Dashboard Entry Point wrapper for WeatherBlend MVP.
Redirects to the new frontend application at frontend/app.py.
"""
import os
import sys

FRONTEND_APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "app.py")

with open(FRONTEND_APP, "r", encoding="utf-8") as f:
    code = compile(f.read(), FRONTEND_APP, 'exec')
    exec(code, {'__name__': '__main__', '__file__': FRONTEND_APP})
