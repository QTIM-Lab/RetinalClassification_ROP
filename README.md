# RetinalClassification_ROP

Retinal Classification Network for Plus Disease in ROP

**Requirements**: 
- Python 3.6.8
- PyTorch 1.0.0 (all versions ≥ 0.9.0 should work)
- CUDA 9.0
- GPU support

**How to Run**:
Run via command line (preferably in a docker with above requirements) using main.py file

`python main.py [action] [data_directory] [csv_file] [model_path]`

Actions: prepare, train, eval, cluster

Data Directory: contains all images

CSV File: has train and test images split with their names, locations, and labels

Model Path: where you want to save the model OR where model is already saved + .epoch#