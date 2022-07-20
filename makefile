include .env

setup_data_folder : clean
	mkdir -pv data/interim
	mkdir -pv data/processed

download_data:
	wget -O "data/raw/user_labels.csv" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${USER_LABEL_DRIVE_FILE_ID}"

download_text:
	wget -O "data/raw/tweets.tar.gz" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${TWEET_TEXT_DRIVE_FILE_ID}"
	tar -xzvf tweets.tar.gz --directory data/raw/

clean : 
	rm -rf data/interim
	rm -rf data/processed

# Preprocess (load and tokenize) tweet text into a HuggingFace dataset
preprocess : 
	python3 -m src.data.make_dataset \
		--csv_path="data/raw/tweets.csv" \
		--processed_dataset_path="data/processed/tweets"

# Set up truncated dataset for testing.
setup_tests :
	mkdir -pv data/testing/raw
	mkdir -pv data/testing/interim
	mkdir -pv data/testing/processed

	head -n 7 data/raw/tweets.csv > data/testing/raw/tweets.csv

run_data_tests:
	python3 -m unittest src.tests

test_preprocess_dataset:
	python3 -m src.data.make_dataset \
		--csv_path="data/testing/raw/tweets.csv" \
		--processed_dataset_path="data/testing/processed/tweets"

test_show_dataset_stats:
	python3 -m src.data.print_dataset_stats \
		--processed_dataset_path="data/testing/processed/tweets"

delete_tests:
	rm -rf data/testing