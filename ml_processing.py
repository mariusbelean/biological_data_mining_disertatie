import psycopg2
from database_connection import create_connection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

def load_data_from_database():
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch protein data from the database
    query = "SELECT gene_name, protein_sequence FROM proteins"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Separate gene names and protein sequences
    gene_names = []
    protein_sequences = []
    for row in rows:
        gene_name, protein_sequence = row
        gene_names.append(gene_name)
        protein_sequences.append(protein_sequence)

    cursor.close()
    conn.close()

    return gene_names, protein_sequences

def preprocess_data(protein_sequences):
    # Use CountVectorizer to convert protein sequences into numerical features
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    X = vectorizer.fit_transform(protein_sequences)
    return X

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model (Support Vector Machine classifier)
    model = SVC()
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)

def main():
    # Load protein data from the database
    gene_names, protein_sequences = load_data_from_database()

    # Perform data preprocessing
    X = preprocess_data(protein_sequences)

    # Define the target variable (e.g., whether a protein is BRCA1 or BRCA2)
    target_gene_names = ["BRCA1_HUMAN"]
    y = [1 if gene_name in target_gene_names else 0 for gene_name in gene_names]

    # Train and evaluate the model
    train_model(X, y)

if __name__ == "__main__":
    main()
