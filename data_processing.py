import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import psycopg2
from database_connection import create_connection
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Function to handle missing values by replacing them with the mean of each column
def handle_missing_values(X):
    for i in range(X.shape[1]):
        column = X[:, i]
        if np.issubdtype(column.dtype, np.number):
            column_means = np.nanmean(column.astype(float))
            nan_mask = np.isnan(column.astype(float))
            column[nan_mask] = column_means
        else:
            column = column.astype(str)
            unique_values, counts = np.unique(column, return_counts=True)
            most_common_value = unique_values[np.argmax(counts)]
            nan_mask = column == ''
            column[nan_mask] = most_common_value
        X[:, i] = column
    return X


# Function to remove duplicates
def remove_duplicates(X, y):
    df = pd.DataFrame(X)
    df['target'] = y
    df.drop_duplicates(inplace=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# Function to balance classes using random oversampling
def balance_classes(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

# Function to encode categorical variables
def encode_categorical_variables(X):
    encoder = LabelEncoder()
    for i in range(X.shape[1]):
        if isinstance(X[0, i], str):
            X[:, i] = encoder.fit_transform(X[:, i])
    return X

# Function to scale numerical features
def scale_numerical_features(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Function to remove constant features
def remove_constant_features(X):
    constant_mask = np.all(X == X[0, :], axis=0)
    X = X[:, ~constant_mask]
    return X

# Function for feature selection using chi-square test
def feature_selection(X, y):
    selector = SelectKBest(chi2, k='all')
    X_selected = selector.fit_transform(X, y)
    return X_selected

# Function for dimensionality reduction using PCA
def dimensionality_reduction(X):
    pca = PCA()
    X_reduced = pca.fit_transform(X)
    return X_reduced

def load_data_from_database():
    # Establish the database connection
    conn = create_connection()

    # Fetch protein data from the database
    cursor = conn.cursor()
    cursor.execute("SELECT protein_sequence, protein_name, gene_name, organism FROM proteins;")
    rows = cursor.fetchall()
    X = np.array([list(row) for row in rows])
    y = X[:, 1]  # Assuming protein_name is the target variable

    # Close the database connection
    conn.close()

    return X, y

def main():
    # Load protein data from the database
    X, y = load_data_from_database()

    # Data processing steps
    print("Original data:")
    print(X)  # Print the original data

    X = handle_missing_values(X)
    print("After handling missing values:")
    print(X)  # Print data after handling missing values

    X, y = remove_duplicates(X, y)
    print("After removing duplicates:")
    print(X)  # Print data after removing duplicates
    print(y)  # Print labels after removing duplicates

    X, y = balance_classes(X, y)
    print("After balancing classes:")
    print(X)  # Print data after balancing classes
    print(y)  # Print labels after balancing classes

    X = encode_categorical_variables(X)
    print("After encoding categorical variables:")
    print(X)  # Print data after encoding categorical variables

    X = scale_numerical_features(X)
    print("After scaling numerical features:")
    print(X)  # Print data after scaling numerical features

    X = remove_constant_features(X)
    print("After removing constant features:")
    print(X)  # Print data after removing constant features

    X = feature_selection(X, y)
    print("After feature selection:")
    print(X)  # Print data after feature selection

    X = dimensionality_reduction(X)
    print("After dimensionality reduction:")
    print(X)  # Print data after dimensionality reduction

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the train set
    print("Train set:")
    print(X_train)

    # Print the test set
    print("Test set:")
    print(X_test)

    #Random Forest machine learning model
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()
