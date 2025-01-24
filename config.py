import pathlib
import os
import demo_project_1 as mlops_project

# Define the root directory of the project by resolving the path of the project's __file__ attribute
ROOT_DIR = pathlib.Path(mlops_project.__file__).resolve().parent

# Define the directory where raw data files are stored
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')

# Define the name of the data file
file_name = 'iris.csv'

# Define the full path to the data file
file_path = os.path.join(DATA_DIR, file_name)

# Define the name of the model file
MODEL_NAME = "iris_model.joblib"

# Define the directory where the model file will be saved
SAVE_MODEL_DIR = os.path.join(ROOT_DIR, 'model')