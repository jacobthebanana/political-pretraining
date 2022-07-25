include .env

setup_data_folder: 
	mkdir -pv data/raw
	mkdir -pv data/external
	mkdir -pv data/interim
	mkdir -pv data/processed
	mkdir -pv data/artifacts

download_data:
	wget -O "data/raw/user_labels.csv" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${USER_LABEL_DRIVE_FILE_ID}"

download_text_csv:
	wget -O "data/raw/tweets.tar.gz" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${TWEET_TEXT_DRIVE_FILE_ID}"
	tar -xzvf tweets.tar.gz --directory data/raw/

download_text_json:
	wget -O "data/raw/tweets.json" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${TWEET_JSON_DRIVE_FILE_ID}"
	sed -i 's/#//g' "data/raw/tweets.json"
	
download_text: download_text_csv download_text_json

clean: 
	rm -rf data/interim
	rm -rf data/processed
	rm -rf data/artifacts

clean_all: clean
	rm -rf data/raw
	rm -rf data/external


setup: clean setup_data_folder download_data download_text

# Preprocess (load and tokenize) tweet text into a HuggingFace dataset
preprocess_csv: 
	python3 -m src.data.make_dataset \
		--source_format=csv \
		--source_path="data/raw/tweets.csv" \
		--processed_dataset_path="data/processed/tweets" \
		--max_seq_length=${max_seq_length} \
		--shard_denominator=${shard_denominator} \
		--base_model_name=${base_model_name}

preprocess_json: 
	python3 -m src.data.make_dataset \
		--source_format=json \
		--source_path="data/raw/tweets.json" \
		--processed_dataset_path="data/processed/tweets" \
		--max_seq_length=${max_seq_length} \
		--shard_denominator=${shard_denominator} \
		--base_model_name=${base_model_name}

# Generate average user embeddings on the given dataset.
embed:
	python3 -m src.models.predict_model \
		--processed_dataset_path="data/processed/tweets" \
		--output_embeddings_json_path="data/artifacts/embeddings-${json_suffix}.json" \
		--base_model_name=${base_model_name} \
		--eval_per_device_batch_size=${eval_per_device_batch_size}

# Set up truncated dataset for testing.
setup_data_tests:
	mkdir -pv data/testing/raw
	mkdir -pv data/testing/interim
	mkdir -pv data/testing/processed

	head -n 302 data/raw/tweets.csv > data/testing/raw/tweets.csv
	head -n 302 data/raw/tweets.json > data/testing/raw/tweets.json

run_data_tests:
	python3 -m unittest src.tests.test_preprocessing_csv
	python3 -m unittest src.tests.test_preprocessing_json

run_model_predict_tests:
	python3 -m unittest src.tests.test_predict_model

preprocess_test_dataset:
	python3 -m src.data.make_dataset \
		--source_path="data/testing/raw/tweets.csv" \
		--processed_dataset_path="data/testing/processed/tweets"

test_show_dataset_stats: preprocess_test_dataset
	python3 -m src.data.print_dataset_stats \
		--processed_dataset_path="data/testing/processed/tweets"

data_test_cleanup:
	rm -rf data/testing

data_tests: setup_data_tests run_data_tests  

model_predict_tests: setup_data_tests preprocess_test_dataset run_model_predict_tests

test_cleanup: data_test_cleanup

test: data_tests model_predict_tests test_cleanup

install_env:
	pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
