"""
Script to inspect Supabase table structures by fetching one row.
"""

import os
import sys
import json
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

def inspect_tables():
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("✗ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    supabase: Client = create_client(supabase_url, supabase_key)
    
    tables = ['User', 'Team', 'Submission']
    
    for table_name in tables:
        print(f"\n--- Inspecting table: {table_name} ---")
        try:
            # Fetch 1 row
            response = supabase.table(table_name).select("*").limit(1).execute()
            
            if response.data and len(response.data) > 0:
                record = response.data[0]
                print(f"Columns found: {list(record.keys())}")
                print(f"Sample record: {json.dumps(record, indent=2)}")
            else:
                print("Table is empty (or no read access). Cannot determine columns from data.")
                print("Trying to insert a dummy record to provoke an error might reveal schema, but let's avoid that for now.")
                
        except Exception as e:
            print(f"Error inspecting {table_name}: {e}")

if __name__ == "__main__":
    inspect_tables()
