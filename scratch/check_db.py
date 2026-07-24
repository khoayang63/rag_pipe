import os
import psycopg2
from psycopg2.extras import RealDictCursor

conn_params = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432"),
    "database": os.environ.get("POSTGRES_DB", "ekb"),
    "user": os.environ.get("POSTGRES_USER", "admin"),
    "password": os.environ.get("POSTGRES_PASSWORD", "123456"),
}

def check():
    try:
        conn = psycopg2.connect(**conn_params)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            chunk_count = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) FROM document_images;")
            image_count = cur.fetchone()["count"]
            
            print(f"--- Database Status ---")
            print(f"Documents: {doc_count}")
            print(f"Document Chunks: {chunk_count}")
            print(f"Document Images: {image_count}")
            
            if doc_count > 0:
                cur.execute("SELECT id, name FROM documents LIMIT 5;")
                print("\n--- Recent Documents ---")
                for row in cur.fetchall():
                    print(f"- ID: {row['id']}, Name: {row['name']}")
                    
            if image_count > 0:
                cur.execute("SELECT id, document_id, image_path, caption FROM document_images LIMIT 5;")
                print("\n--- Recent Images ---")
                for row in cur.fetchall():
                    print(f"- ID: {row['id']}, Doc ID: {row['document_id']}, Path: {row['image_path']}, Caption: {row['caption']}")
            
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    check()
