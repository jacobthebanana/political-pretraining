setup_data_folder : clean
	mkdir data/raw
	mkdir data/external
	mkdir data/interim
	mkdir data/processed

clean : 
	rm -rf data/interim
	rm -rf data/processed

preprocess : 
	