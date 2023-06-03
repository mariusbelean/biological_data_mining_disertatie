import glob
from database_connection import create_connection

class Protein_Unlabelled:
    def __init__(self, gene_name, organism, protein_sequence):
        self.gene_name = gene_name
        self.organism = organism
        self.protein_sequence = protein_sequence

def process_fasta_file(fasta_file_path):
    fasta_file = open(fasta_file_path, "r")
    sequences = []
    current_sequence = None

    for line in fasta_file:
        line = line.strip()
        if line.startswith(">"):
            # If there was a previous sequence, add it to the list
            if current_sequence is not None:
                sequences.append(current_sequence)

            header = line[1:]
            metadata = header.split("|")
            gene_name = ""
            organism = ""

            if len(metadata) >= 3:
                gene_name = metadata[2].split(" ")[0]

            if len(metadata) >= 4:
                organism = metadata[2][len(gene_name)+1:].strip()

            current_sequence = Protein_Unlabelled(gene_name, organism, "")
        else:
            # Append the line to the current sequence's protein_sequence
            if current_sequence is not None:
                current_sequence.protein_sequence += line

    # Add the last sequence to the list
    if current_sequence is not None:
        sequences.append(current_sequence)

    fasta_file.close()

    return sequences


def insert_unlabelled_protein_data(connection, proteins):
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO new_prediction_dataset (gene_name, organism, protein_sequence)
    VALUES (%s, %s, %s)
    """

    for protein in proteins:
        data = (protein.gene_name, protein.organism, protein.protein_sequence)
        cursor.execute(insert_query, data)

    connection.commit()
    cursor.close()


# Get a list of all FASTA files in the dataset folder
fasta_files = glob.glob("prediction_dataset/*.fasta")
sequences = []

for fasta_file in fasta_files:
    sequences.extend(process_fasta_file(fasta_file))

# Print the extracted protein information
for sequence in sequences:
    print("Gene Name:", sequence.gene_name)
    print("Organism:", sequence.organism)
    print("Protein Sequence:", sequence.protein_sequence)
    print()

# Establish the database connection
conn = create_connection()

# Insert the protein data into the database
insert_unlabelled_protein_data(conn, sequences)

# Close the database connection
conn.close()
