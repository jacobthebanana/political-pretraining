# Setup
## Create Data Folder
After inflating, this project would require around 100 GB of free space for its datasets. 
Consider creating a symbolic link from your scratch folder to the data/ folder under the project root folder.

For example:
```bash
mkdir $SCRATCH/tweets
ln -s $SCRATCH/tweets data
```

## Download Data
Contact [Jacob](mailto:jacob.mila-complex-data-lab-github.handle@tianshome.com) for a list of file IDs. Save these in `.env` under the project root folder (same folder as the `makefile`.) Run the following from the project root folder to retrieve data:
```bash
make setup
```

## Preprocess Data
Note that by default, HuggingFace Datasets uses `~/.cache` as the cache folder. For performance reasons, consider setting the env var `HF_DATASETS_CACHE` to a location on the scratch drive.

Run:
```bash
make preprocess
```

# Contributing
## Unittests
After finishing steps in the "setup" section, run the following to unit test the code: 
```bash
make test
```