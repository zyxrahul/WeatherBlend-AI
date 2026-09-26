.PHONY: all data baselines stacking gating backend frontend demo clean

all: demo

data:
	python backend/src/generate_data.py

baselines:
	python backend/src/baselines.py

stacking:
	python backend/src/stacking_model.py

gating:
	python backend/src/gating_model.py

backend:
	python run_backend.py

frontend:
	streamlit run frontend/app.py

demo: data baselines stacking gating frontend

clean:
	rm -rf backend/data/*.parquet backend/data/*.csv backend/models/*.joblib
