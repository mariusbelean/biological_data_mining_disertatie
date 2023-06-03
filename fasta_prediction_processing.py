import numpy as np
import psycopg2  # Assuming you're using PostgreSQL
from data_processing import load_model
from database_connection import create_connection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def make_predictions(data):
    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Load the trained model
    model = load_model('trained_model/trained_model.pickle')
    
    print("Estimators: ",model.n_estimators)  # Print the number of decision trees in the random forest
    print("Feature importances: ",model.feature_importances_)  # Print the feature importances determined by the model
    print("Classes", model.classes_)  # Print the unique classes or labels in the target variable

    # Reshape the data
    reshaped_data = reshape_data(preprocessed_data)

    # Make predictions
    predictions = model.predict(reshaped_data)

    return predictions


def preprocess_data(data):
    # Separate the features (gene name, organism, protein sequence)
    gene_names = [str(record[0]) for record in data]
    organisms = [str(record[1]) for record in data]
    protein_sequences = [str(record[2]).upper() for record in data]

    # Create a label encoder for each feature
    label_encoder = LabelEncoder()

    # Encode and reshape the gene names
    gene_name_encoded = label_encoder.fit_transform(gene_names)
    reshaped_gene_name = gene_name_encoded.reshape(-1, 1)

    # Encode and reshape the organisms
    organism_encoded = label_encoder.fit_transform(organisms)
    reshaped_organism = organism_encoded.reshape(-1, 1)

    # Encode and reshape the protein sequences
    protein_sequence_encoded = label_encoder.fit_transform(protein_sequences)
    reshaped_protein_sequence = protein_sequence_encoded.reshape(-1, 1)

    # Concatenate the encoded features
    reshaped_data = np.concatenate((reshaped_gene_name, reshaped_organism, reshaped_protein_sequence), axis=1)

    return reshaped_data


def reshape_data(data):
    # Separate the features (gene name, organism, protein sequence)
    gene_names = [str(record[0]) for record in data]
    organisms = [str(record[1]) for record in data]
    protein_sequences = [str(record[2]).upper() for record in data]

    # Create a label encoder for each feature
    label_encoder = LabelEncoder()

    # Encode the gene names
    encoded_gene_names = label_encoder.fit_transform(gene_names)

    # Encode the organisms
    encoded_organisms = label_encoder.fit_transform(organisms)

    # Encode the protein sequences
    encoded_protein_sequences = label_encoder.fit_transform(protein_sequences)

    # Create a reshaped_data array with three features
    reshaped_data = np.column_stack((encoded_gene_names, encoded_organisms, encoded_protein_sequences))

    return reshaped_data



def fetch_data_from_database():

    connection = create_connection()
    # Create a cursor object to execute queries
    cursor = connection.cursor()

    # Execute a query to fetch the data from the new_prediction_dataset table
    cursor.execute('SELECT gene_name, organism, protein_sequence FROM new_prediction_dataset')

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
    # with open("processed_predictions6.txt", "w") as file:
    #     for prediction in predictions:
    #         file.write(str(prediction) + "\n")

if __name__ == '__main__':
    main()
