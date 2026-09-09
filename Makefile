.PHONY: data

data:
	python data/prepare_positions.py
	python data/label_positions.py
	python data/create_features.py
	python data/data_validation.py
	python data/create_split.py