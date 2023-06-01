import psycopg2

def create_connection():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="ABD_PF",
        user="postgres",
        password="postgres"
)
    return conn
