import numpy as np
import psycopg2
from data_processing import load_model
from database_connection import create_connection
from sklearn.preprocessing import LabelEncoder

import numpy as np

def make_predictions(data):
    # Load the trained model
    model = load_model('trained_model/trained_model.pickle')

    print("Estimators: ", model.n_estimators)  # Print the number of decision trees in the random forest
    print("Feature importances: ", model.feature_importances_)  # Print the feature importances determined by the model
    print("Classes", model.classes_)  # Print the unique classes or labels in the target variable

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Make predictions
    predictions = model.predict(preprocessed_data)

    # Generate random predictions based on model.classes_
    random_predictions = np.random.choice(model.classes_, size=len(predictions))

    # Retrieve gene_name, organism, and protein_sequence from the input data
    gene_names = [record[0] for record in data]
    organisms = [record[1] for record in data]
    protein_sequences = [record[2] for record in data]

    # Combine the input data with the predicted protein names
    output = list(zip(gene_names, organisms, protein_sequences, random_predictions))

    return output


def preprocess_data(data):
    # Separate the features (gene name, organism, protein sequence)
    gene_names = [str(record[0]) for record in data]
    organisms = [str(record[1]) for record in data]
    protein_sequences = [str(record[2]).upper() for record in data]

    # Create a label encoder for each feature
    label_encoder = LabelEncoder()

    # Fit the label encoder on the training data
    label_encoder.fit(gene_names)

    # Encode the gene names
    encoded_gene_names = label_encoder.transform(gene_names)

    # Fit the label encoder on the training data
    label_encoder.fit(organisms)

    # Encode the organisms
    encoded_organisms = label_encoder.transform(organisms)

    # Fit the label encoder on the training data
    label_encoder.fit(protein_sequences)

    # Encode the protein sequences
    encoded_protein_sequences = label_encoder.transform(protein_sequences)

    # Create a reshaped_data array with three features
    reshaped_data = np.column_stack((encoded_gene_names, encoded_organisms, encoded_protein_sequences))

    return reshaped_data


def fetch_data_from_database():
    connection = create_connection()
    # Create a cursor object to execute queries
    cursor = connection.cursor()

    # Execute a query to fetch the data from the new_prediction_dataset table
    cursor.execute('SELECT gene_name, organism, protein_sequence   FROM new_prediction_dataset')

    # Fetch all rows of the result
    data = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return data


def main():
    # Fetch the data from the new_prediction_dataset table
    data = fetch_data_from_database()

    # Make predictions using the fetched data
    predictions = make_predictions(data)

    # Print the predictions
    print(predictions)

    # Save the processed predictions to a file
    # with open("processed_predictions.txt", "w") as file:
    #     for prediction in predictions:
    #         file.write(str(prediction) + "\n")


if __name__ == '__main__':
    main()
