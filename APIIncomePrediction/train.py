# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle


# Loading cleaned data
raw_df = pd.read_csv(r'Projects\APIIncomePrediction\processed.csv')

# Removing Education column since it is already encoded to Education_level
raw_df = raw_df.drop('Education', axis=1)

# Separating numerical and categorical columns
num_columns = raw_df.select_dtypes(include=['int64']).columns.tolist()
num_columns.remove('Income')
cat_columns = raw_df.select_dtypes(include=['object']).columns.tolist()

# Separating Features and Target
X = raw_df.drop('Income', axis=1)
y = raw_df[['Income']].values.flatten()

# Encoding and Scaling
transformer = ColumnTransformer([('cat', OneHotEncoder(drop='first', sparse_output=False), cat_columns), 
                                 ('num', StandardScaler(), num_columns)], remainder='passthrough')
X = transformer.fit_transform(X)

# Saving the transformer
with open(r'Projects\APIIncomePrediction\transformer.pkl', "wb") as f:
    pickle.dump(transformer, f)

# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# LOGISTIC REGRESSION MODEL
print("\nLOGISTIC MODEL:")

# Train model
model_log = LogisticRegression(class_weight='balanced', random_state=42)
model_log.fit(X_train, y_train)

# Evaluate model
y_pred = model_log.predict(X_test)
y_prob = model_log.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy * 100:.2f}")

f1 = f1_score(y_test, y_pred)
print(f"F1 score: {f1 * 100:.2f}")

roc_auc = roc_auc_score(y_test, y_prob[:, 1])
print(f"AUC Score: {roc_auc * 100:.2f}")


# RANDOM FOREST MODEL
print("\nRANDOM FOREST MODEL:")

# Train model
model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate model
y_pred = model_rf.predict(X_test)
y_prob = model_rf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy * 100:.2f}")

f1 = f1_score(y_test, y_pred)
print(f"F1 score: {f1 * 100:.2f}")

roc_auc = roc_auc_score(y_test, y_prob[:, 1])
print(f"AUC Score: {roc_auc * 100:.2f}")


# Saving best model
print("\nHence, Random Forest model is the best model.")
with open(r'Projects\APIIncomePrediction\model.pkl', "wb") as f:
    pickle.dump(model_rf, f)

