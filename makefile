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

# Generate filtered label file where classifier label must be non-empty.
generate_filtered_label_file:
	python3 -m src.data.filter_label_file \
		--raw_label_path="data/raw/user_labels.csv" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--label_id_to_label_text_path="data/interim/label_id_to_label_text.json"

# Preprocess (load and tokenize) tweet text into a HuggingFace dataset
preprocess_csv: generate_filtered_label_file
	python3 -m src.data.make_dataset \
		--source_format=csv \
		--source_path="data/raw/tweets.csv" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--require_labels=${require_labels} \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}" \
		--processed_lookup_by_uid_json_path="data/processed/tweets-${processed_dataset_suffix}/lookup_by_uid.json" \
		--max_seq_length=${max_seq_length} \
		--shard_denominator=${shard_denominator} \
		--base_model_name=${base_model_name} \
		--enable_indexing=${enable_indexing} \
		--rerun_tokenization=${rerun_tokenization} \
		--per_user_concatenation=${per_user_concatenation} \
		--concatenation_delimiter=${concatenation_delimiter} \
		--num_procs=${num_procs}

preprocess_json: generate_filtered_label_file
	python3 -m src.data.make_dataset \
		--source_format=json \
		--source_path="data/raw/tweets.json" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--require_labels=${require_labels} \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}" \
		--processed_lookup_by_uid_json_path="data/processed/tweets-${processed_dataset_suffix}/lookup_by_uid.json" \
		--max_seq_length=${max_seq_length} \
		--shard_denominator=${shard_denominator} \
		--base_model_name=${base_model_name} \
		--enable_indexing=${enable_indexing} \
		--rerun_tokenization=${rerun_tokenization} \
		--per_user_concatenation=${per_user_concatenation} \
		--concatenation_delimiter=${concatenation_delimiter} \
		--num_procs=${num_procs}

show_dataset_stats: 
	python3 -m src.data.print_dataset_stats \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}"

train:
	python3 -m src.models.train_model \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}" \
		--processed_lookup_by_uid_json_path="data/processed/tweets-${processed_dataset_suffix}/lookup_by_uid.json" \
		--base_model_name=${base_model_name} \
		--triplet_threshold=${triplet_threshold} \
		--train_per_device_batch_size=${train_per_device_batch_size} \
		--eval_per_device_batch_size=${eval_per_device_batch_size} \
		--pooling_strategy=${pooling_strategy} \
		--model_output_path=${model_output_path} \
		--save_every_num_batches=${save_every_num_batches} \
		--distance_function=${distance_function} \
		--enable_masking=${enable_masking} \
		--num_epochs=${num_epochs}

# Generate average user embeddings on the given dataset.
embed:
	python3 -m src.models.predict_model \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}" \
		--output_embeddings_json_path="data/artifacts/embeddings-${json_suffix}.json" \
		--base_model_name=${base_model_name} \
		--eval_per_device_batch_size=${eval_per_device_batch_size} \
		--pooling_strategy=${pooling_strategy}

# Set up truncated dataset for testing.
setup_data_tests: generate_filtered_label_file
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

run_model_train_tests:
	python3 -m unittest src.tests.test_train_model

preprocess_test_dataset:
	head -n 302 data/raw/tweets.csv > data/testing/raw/tweets.csv
	python3 -m src.data.make_dataset \
		--base_model_name=${unittest_base_model_name} \
		--source_path="data/testing/raw/tweets.csv" \
		--per_user_concatenation=1 \
		--enable_indexing=1 \
		--processed_dataset_path="data/testing/processed/tweets" \
		--processed_lookup_by_uid_json_path="data/testing/processed/tweets/lookup_by_uid.json" 

test_show_dataset_stats: preprocess_test_dataset
	python3 -m src.data.print_dataset_stats \
		--processed_dataset_path="data/testing/processed/tweets"

data_test_cleanup:
	rm -rf data/testing

data_tests: setup_data_tests run_data_tests  

model_predict_tests: setup_data_tests preprocess_test_dataset run_model_predict_tests run_model_train_tests

test_cleanup: data_test_cleanup

test: data_tests model_predict_tests test_cleanup

install_env:
	pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
