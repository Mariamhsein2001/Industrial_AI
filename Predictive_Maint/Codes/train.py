import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
import os

def preprocess_data(df):
    #change  columns to float
    for column in df.columns:
      try:
          df[column]=df[column].astype(float)
      except:
          pass
    # create mulitclass column
    df['Machine failure'] = 0

    # Update the 'Machine failure' column based on conditions
    df.loc[df['TWF'] == 1, 'Machine failure'] = 1
    df.loc[df['HDF'] == 1, 'Machine failure'] = 2
    df.loc[df['PWF'] == 1, 'Machine failure'] = 3
    df.loc[df['OSF'] == 1, 'Machine failure'] = 4
    df.loc[df['RNF'] == 1, 'Machine failure'] = 5

    #drop columns
    df.drop(['TWF','HDF','PWF','OSF','RNF','UDI','Product ID'],axis=1,inplace=True)

    # Handle missing values using mean imputation for numerical columns
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=['float64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64']))

    df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    df['Type_L'] = df['Type_L'].astype(int)
    df['Type_M'] = df['Type_M'].astype(int)
    
     # Perform feature engineering
    df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    df['Temperature difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # Rename columns to remove problematic characters
    df.rename(columns=lambda x: x.replace('[', '_').replace(']', '_').replace('<', '_'), inplace=True)
    return df

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

def train_models(X_train, y_train, models, param_grids, cv=5):
    best_models = {}
    for name, model in models.items():
        try:
            grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='recall_weighted')
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
        except Exception as e:
            print(f"An error occurred while fitting {name}: {e}")
            best_models[name] = None
    return best_models

def evaluate_models(models, X_test, y_test):
    model_scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        model_scores[name] = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
    return model_scores


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
# Load data
df = pd.read_csv(os.path.join(data_dir, 'ai4i2020.csv'))

# Preprocess data
df = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = split_data(df, 'Machine failure')
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Common ML classifiers
models = {
    'XGBoost': XGBClassifier()
}

# Define a set of parameters for each model
param_grids = {
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.5]
    }
}

# Train models
best_models = train_models(X_train, y_train, models, param_grids)

# Evaluate models
model_scores = evaluate_models(best_models, X_test, y_test)
# Display results
best_model_name = max(model_scores, key=lambda name: model_scores[name]['test_recall'])

for name, scores in model_scores.items():
    if name == best_model_name:
        print(f"Best Model: {name}")
        print(f"Test Accuracy: {scores['test_accuracy']:.4f}")
        print(f"Test Precision: {scores['test_precision']:.4f}")
        print(f"Test Recall: {scores['test_recall']:.4f}")
        print(f"Test F1 Score: {scores['test_f1']:.4f}")
        print(f"Best Parameters: {best_models[name].get_params()}\n")  # Display best parameters

best_model = best_models[best_model_name]
best_model.save_model('xgb_model.json')