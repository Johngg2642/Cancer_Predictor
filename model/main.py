import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle 


def create_model(data):
    # Split the data into features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train and fit the model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    """
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    """

    # test the model
    y_pred = model.predict(X_test)
    print("The Accuracy of the model is: ", accuracy_score(y_test, y_pred) * 100)
    print("Classification Report: \n", classification_report(y_test, y_pred))


    return model, scaler

 
def get_clean_data():
    # Read the CSV file into a DataFrame
    data = pd.read_csv("data/data.csv")

    # Drop the 'Unnamed: 32' and id columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    # export the model and scaler
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

 

if __name__ == "__main__":
    main() 