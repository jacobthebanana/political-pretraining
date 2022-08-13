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

download_politician_json:
	wget -O "data/raw/politician_tweets.jsonl" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${POLITICIAN_TWEET_FILE_ID}"
	wget -O "data/raw/politician_labels.jsonl" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${POLITICIAN_LABEL_FILE_ID}"

download_true_labels:
	wget -O "data/raw/screen_names.tsv" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${SCREEN_NAMES_TSV_FILE_ID}"
	wget -O "data/raw/true_labels.jsonl" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${TRUE_USER_LABELS_FILE_ID}"

download_test_uids:
	wget -O "data/raw/test_uids.csv" "${GOOGLE_DRIVE_EXPORT_LINK_PREFIX}&id=${TEST_SUBSET_USER_FILE_ID}"

download_text: download_text_csv download_text_json

clean: 
	rm -rf data/interim
	rm -rf data/processed
	rm -rf data/artifacts

clean_all: clean
	rm -rf data/raw
	rm -rf data/external


setup: clean setup_data_folder download_data download_text

merge_label_files:
	python3 -m src.data.merge_label_files \
		--raw_true_label_jsonl_path="data/raw/true_labels.jsonl" \
		--scree_name_to_uid_tsv_path="data/raw/screen_names.tsv" \
		--processed_true_label_path="data/interim/true_labels.csv"


# Verify that all test labels are available as true labels.
select_test_uids: download_true_labels download_test_uids merge_label_files
	python3 -m src.data.slice_labels \
		"data/raw/test_uids.csv" \
		"data/interim/true_labels.csv" \
		"data/interim/test_labels.csv"

# Generate filtered label file where classifier label must be non-empty.
generate_filtered_label_file:
	python3 -m src.data.filter_label_file \
		--raw_label_path="data/raw/user_labels.csv" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--train_filtered_label_path="data/interim/train_filtered_user_labels.csv" \
		--validation_filtered_label_path="data/interim/validation_filtered_user_labels.csv" \
		--test_filtered_label_path="data/interim/test_filtered_user_labels.csv" \
		--label_id_to_label_text_path="data/interim/label_id_to_label_text.json" \
		--train_test_split_prng_seed=${train_test_split_prng_seed} \
		--test_ratio=${test_ratio} \
		--validation_ratio=${validation_ratio}

# Generate filtered label file with true labels as test set.
generate_filtered_label_file_with_true_labels:
	python3 -m src.data.filter_label_file \
		--raw_label_path="data/raw/user_labels.csv" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--use_true_label_for_test_split=1 \
		--processed_true_label_path="data/interim/true_labels.csv" \
		--train_filtered_label_path="data/interim/train_filtered_user_labels.csv" \
		--validation_filtered_label_path="data/interim/validation_filtered_user_labels.csv" \
		--test_filtered_label_path="data/interim/test_filtered_user_labels.csv" \
		--label_id_to_label_text_path="data/interim/label_id_to_label_text.json" \
		--train_test_split_prng_seed=${train_test_split_prng_seed} \
		--validation_ratio=${validation_ratio}

# Generate filtered label file with test true labels.
generate_filtered_label_file_with_test_labels: select_test_uids
	python3 -m src.data.filter_label_file \
		--raw_label_path="data/raw/user_labels.csv" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--use_true_label_for_test_split=1 \
		--processed_true_label_path="data/interim/test_labels.csv" \
		--train_filtered_label_path="data/interim/train_filtered_user_labels.csv" \
		--validation_filtered_label_path="data/interim/validation_filtered_user_labels.csv" \
		--test_filtered_label_path="data/interim/test_filtered_user_labels.csv" \
		--label_id_to_label_text_path="data/interim/label_id_to_label_text.json" \
		--train_test_split_prng_seed=${train_test_split_prng_seed} \
		--validation_ratio=${validation_ratio}


# Preprocess (load and tokenize) tweet text into a HuggingFace dataset
preprocess_csv:
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

preprocess_json:
	python3 -m src.data.make_dataset \
		--source_format=json \
		--source_path="data/raw/tweets.json" \
		--filtered_label_path="data/interim/filtered_user_labels.csv" \
		--require_labels=${require_labels} \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}" \
		--processed_lookup_by_uid_json_path="data/processed/tweets-${processed_dataset_suffix}/lookup_by_uid.json" \
		--train_filtered_label_path="data/interim/train_filtered_user_labels.csv" \
		--validation_filtered_label_path="data/interim/validation_filtered_user_labels.csv" \
		--test_filtered_label_path="data/interim/test_filtered_user_labels.csv" \
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
		--wandb_entity=${wandb_entity} \
		--wandb_project=${wandb_project} \
		--num_epochs=${num_epochs}

train_cross_entropy:
	python3 -m src.models.train_model_cross_entropy \
		--processed_dataset_path="data/processed/tweets-${processed_dataset_suffix}" \
		--base_model_name=${base_model_name} \
		--train_per_device_batch_size=${train_per_device_batch_size} \
		--eval_per_device_batch_size=${eval_per_device_batch_size} \
		--model_output_path=${model_output_path} \
		--save_every_num_batches=${save_every_num_batches} \
		--weight_decay=${weight_decay} \
		--test_ratio=${test_ratio} \
		--eval_every_num_batches=${eval_every_num_batches} \
		--wandb_entity=${wandb_entity} \
		--wandb_project=${wandb_project} \
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
		--processed_dataset_path="data/testing/processed/tweets" \
		--processed_lookup_by_uid_json_path="data/testing/processed/tweets/lookup_by_uid.json" 

preprocess_test_json_dataset:
	cp data/interim/train_filtered_user_labels.csv data/interim/test_filtered_user_labels.csv
	python3 -m src.data.make_dataset \
		--base_model_name=${unittest_base_model_name} \
		--source_path="data/testing/raw/tweets.json" \
		--source_format=json \
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
