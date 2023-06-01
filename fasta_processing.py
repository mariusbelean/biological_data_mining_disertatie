import glob
from database_connection import create_connection
from insert_data import insert_protein_data

class Protein:
    def __init__(self, protein_id, gene_name, organism, protein_name, protein_sequence):
        self.protein_id = protein_id
        self.gene_name = gene_name
        self.protein_name = protein_name
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
            protein_id = metadata[1]
            gene_name = ""
            protein_name = ""
            organism = ""

            if len(metadata) >= 3:
                gene_name = metadata[2].split(" ")[0]

            if len(metadata) >= 4:
                organism = metadata[2][len(gene_name)+1:].strip()
                protein_name = metadata[3].split("=")[1]

            current_sequence = Protein(protein_id, gene_name, protein_name, organism, "")
        else:
            # Append the line to the current sequence's protein_sequence
            if current_sequence is not None:
                current_sequence.protein_sequence += line

    # Add the last sequence to the list
    if current_sequence is not None:
        sequences.append(current_sequence)

    fasta_file.close()

    return sequences

# Get a list of all FASTA files in the dataset folder
fasta_files = glob.glob("dataset/*.fasta")
sequences = []

for fasta_file in fasta_files:
    sequences.extend(process_fasta_file(fasta_file))

# Print the extracted protein information
for sequence in sequences:
    print("Protein ID:", sequence.protein_id)
    print("Gene Name:", sequence.gene_name)
    print("Protein Name:", sequence.protein_name)
    print("Organism:", sequence.organism)
    print("Protein Sequence:", sequence.protein_sequence)
    print()

# Establish the database connection
conn = create_connection()

# Insert the protein data into the database
insert_protein_data(conn, sequences)

# Close the database connection
conn.close()
