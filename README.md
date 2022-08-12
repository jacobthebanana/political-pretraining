# Setup

## Create Data Folder

After inflating, this project would require around 100 GB of free space for its datasets.
Consider creating a symbolic link from your scratch folder to the `data/` folder under the project root folder.

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
make generate_filtered_label_file_with_true_labels
make preprocess_json
```

## Train Model

Pre-train model with the triplet loss objective:

```bash
make train
```

Fine-tune model with to predict affiliation from text (with the cross-entropy objective function):

```bash
make train_cross_entropy
```

## Generate Embeddings

Generate per-user text embeddings with the given HuggingFace Transformer model:
```bash
make embed
```


# Contributing

## Unittests

After finishing steps in the "setup" section, run the following to unit test the code:

```bash
make test
```
