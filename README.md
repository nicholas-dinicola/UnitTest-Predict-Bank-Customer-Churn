# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- Dataset available in Kaggle: https://www.kaggle.com/sakshigoyal7/credit-card-customers
## Project Description
The project aims to build a machine learning model able to predict if a bank customer will churn or not, utilising
a production-quality code as well as through performing unit tests. 
In order to better understand the problem, the required exploratory data analysis has been performed, understanding
the dataset in a clearer way, generating uni-variate and multi-variate plots. Afterwards, the necessary feature 
engineering preprocessing steps have been completed and two models were trained, a Random Forest (using a Grid 
Search Cross-Validation) and a Logistic Regression. 
We firstly completed all the above-mentioned analysis in an interactive notebooks. 
Afterwards, code have been refactored and cleaned, creating a modular code in a python library able to 
undertake all the analysis done in the notebook. 
Furthermore, the python library has been tested with logging and uni tests. The results all the tests as well as the 
trained model, model scores and EDA graphs have been saved in specific folders.
Lastly, all the code has been cleaned and formatted following the PEP8 standards. 

## Running Files

*Install Packages:*
- pip install -r requirements.txt 

The iPython notebook documents all the analysis steps. 

Running *$ python3 churn_library.py* in the command line you will read in the dataset, perform EDA and Feature 
Engineering as well as training the Random Forest and Logistic Regression algorithms. The results of the EDA are visible 
in the directory *images/eda*, while the models' classification report are stored in directory *images/results*. 

The PEP8 score is visible by running *$ pylint churn_library.py* (it scored above 8/10). 

Running *"$ python3 churn_script_logging_and_tests.py"* you will perform uni tests on the *churn_library.py* file. 
The results of each unit tests is stored in the *churn_library.log* file in the *logs* directory. 

The Also this file follows the PEP8 standard, the score is visible by running 
*$ pylint churn_script_logging_and_tests.py* (it scored above 8/10). 

The peakle file containing the two trained models are in rhe *models* directory. 

Lastly, the code has been cleaned and formatted also by running *$ autopep8 --in-place 
--aggressive --aggressive **name_of_the_file.py***. 







