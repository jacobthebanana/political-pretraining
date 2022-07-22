include .env

setup_data_folder: clean
	mkdir -pv data/raw
	mkdir -pv data/external
	mkdir -pv data/interim
	mkdir -pv data/processed

download_data:
	wget -O "data/raw/user_labels.csv" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${USER_LABEL_DRIVE_FILE_ID}"

download_text:
	wget -O "data/raw/tweets.tar.gz" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${TWEET_TEXT_DRIVE_FILE_ID}"
	tar -xzvf tweets.tar.gz --directory data/raw/

clean: 
	rm -rf data/interim
	rm -rf data/processed

clean_all: clean
	rm -rf data/raw
	rm -rf data/external


setup: clean setup_data_folder download_data download_text

# Preprocess (load and tokenize) tweet text into a HuggingFace dataset
preprocess: 
	python3 -m src.data.make_dataset \
		--csv_path="data/raw/tweets.csv" \
		--processed_dataset_path="data/processed/tweets"

# Set up truncated dataset for testing.
setup_data_tests:
	mkdir -pv data/testing/raw
	mkdir -pv data/testing/interim
	mkdir -pv data/testing/processed

	head -n 302 data/raw/tweets.csv > data/testing/raw/tweets.csv

run_data_tests:
	python3 -m unittest src.tests.test_data_preprocessing

run_model_predict_tests:
	python3 -m unittest src.tests.test_predict_model

preprocess_test_dataset:
	python3 -m src.data.make_dataset \
		--csv_path="data/testing/raw/tweets.csv" \
		--processed_dataset_path="data/testing/processed/tweets"

test_show_dataset_stats: preprocess_test_dataset
	python3 -m src.data.print_dataset_stats \
		--processed_dataset_path="data/testing/processed/tweets"

data_test_cleanup:
	rm -rf data/testing

data_tests: setup_data_tests run_data_tests test_show_dataset_stats  

model_predict_tests: setup_data_tests preprocess_test_dataset run_model_predict_tests

test_cleanup: data_test_cleanup

test: data_tests model_predict_tests test_cleanup