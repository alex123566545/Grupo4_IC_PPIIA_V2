import psycopg2
import os
import sys

from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host="aws-0-us-west-2.pooler.supabase.com",
        port=5432,
        database="postgres",
        user="postgres.cuxppijddpiuaxwyswfb",
        password="alex20151615665451",
        sslmode="require"
    )