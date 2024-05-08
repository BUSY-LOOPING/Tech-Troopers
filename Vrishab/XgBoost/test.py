#code playground file
from helpers.validation_helper import parseAndValidateStringToIntRanges

import os
print(os.getcwd())

path = os.path.join("..", 'Data')
print(path)

# Get the current working directory
current_dir = os.getcwd()

# Navigate to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "Data"))

print(parent_dir)

import config
config.createConfig()