import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host="aws-0-us-west-2.pooler.supabase.com",
        port=5432,
        database="postgres",
        user="postgres.cuxppijddpiuaxwyswfb",
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        sslmode="require"
    )
