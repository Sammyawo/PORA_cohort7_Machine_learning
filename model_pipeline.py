import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load the dataset
dataset = pd.read_csv("Data/insurance.csv")

# Handling categorical variables
dataset.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
dataset['Geography'].unique()
geography_dummies = pd.get_dummies(['Geography'], drop_first=True)
dataset = pd.concat([geography_dummies, dataset], axis=1)
dataset.drop(['Geography'], axis=1, inplace=True)

# Gender
dataset['Gender'] = dataset['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

# Creating the Training Set and Test Set
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build and train the model
model = xgb.XGBClassifier(max_depth=8, learning_rate=0.3, n_estimators=100)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)

joblib.dump(model, 'model/model.pkl')
print("Model saved as 'model.pkl'")

#if __name__ == "__main__":
#    print(dataset)