# Income Prediction ML model and API

In this project, I analyse the 'Adult Income Prediction' dataset from Kaggle and then build two machine learning models to classify the individuals with high or low incomes. I then build an API to do batch predictions using the best model.

#### Data source:
* https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset/data

#### Contains:
1. income.csv - Raw dataset
2. Analysis.ipynb - Data preprocessing and analysis, saves cleaned dataset to processed.csv
3. processed.csv - Cleaned dataset
4. train.py - ML model building and training using processed.csv, saves transformer.pkl and model.pkl
5. transformer.pkl - Transformer used for encoding and scaling
6. model.pkl - Trained Random Forest model (could not be uploaded due to size limitations)
7. app.py - API
8. batch_predict.py - Automation script for batch predictions
9. input.csv - sample input data for batch predictions
10. output.csv - output of the batch predictions
11. README.md - For info

#### Model Details:
1. Uses processed.csv for training
2. Does One Hot Encoding and Feature Scaling, and saves the transformer
3. Build Logistic and Random Forest Model for binary classification
4. Random Forest model had better metrics.

#### How to use:
1. Make sure the files are in the appropriate directory. If any FileNotFound Error occurs, change the corresponding paths in the files.
2. Run Analysis.py, this will create processed.csv.
3. Run train.py to train and save the model.
4. On the command prompt, change to appropriate directory and run 'python app.py'.
5. On the command prompt, run 'python batch_predict.py input.csv output.csv' to use the sample input.csv for batch predictions.
6. The predictions should be saved in output.csv.

PS: 'model.pkl' could not be uploaded due to size limitations. It can be obtained by running 'train.py'.

