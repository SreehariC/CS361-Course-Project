# Running GMMHMM python Script


## Prerequisites

- Python installed on your system. You can download Python from the [official website](https://www.python.org/downloads/).
- `pip` package manager installed. It usually comes pre-installed with Python. You can check if it's installed by running `pip --version` in your command line interface.

## Installation

1. Download the project repository to your local machine.
2. Navigate to the directory containing the Python script and the requirements.txt file.
3. Install the required dependencies using pip. Ensure you have pip installed and it's up to date.
```
pip install -r requirements.txt
```
## Running the Script
Once you have installed the dependencies, you can run the Python script. Navigate to the directory containing the script and execute it using the Python interpreter.

```
 python GMMHMM.py
```
## Additional Notes
Once the script finishes running you can view the confusion matrix and it will also print metrics such as accuracy, precision, recall and f1 score. Here inorder to get metrics for all classes we have used a macro average approach. Furthermore you can also edit variables
```
#Hyperparameters
number_of_states = 4
number_of_gaussians = 3
sound_path='../Dataset/HindiDigits/'
```  
To try and get different results here sound path is path to training data
