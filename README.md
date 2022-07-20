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
Contact [Jacob](jacob.mila-complex-data-lab-github.handle@tianshome.com) for a list of file IDs. Save these in `.env` in the project root folder (same folder as the `makefile`.) Run the following from the project root folder to retrieve data:
```bash
make setup
```
