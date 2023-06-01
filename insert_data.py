import psycopg2

def insert_protein_data(connection, proteins):
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO proteins (protein_id, gene_name, protein_name, organism, protein_sequence)
    VALUES (%s, %s, %s, %s, %s)
    """

    for protein in proteins:
        data = (protein.protein_id, protein.gene_name, protein.protein_name, protein.organism, protein.protein_sequence)
        cursor.execute(insert_query, data)

    connection.commit()
    cursor.close()
