.PHONY: all data baselines stacking gating dashboard demo clean

all: demo

data:
	python src/generate_data.py

baselines:
	python src/baselines.py

stacking:
	python src/stacking_model.py

gating:
	python src/gating_model.py

dashboard:
	streamlit run dashboard/app.py

demo: data baselines stacking gating dashboard

clean:
	rm -rf data/*.parquet data/*.csv models/*.joblib
