"""
Script to clear User, Team, and Submission tables in Supabase.
WARNING: This will delete ALL data in these tables.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

def clear_tables():
    supabase_url = os.getenv('SUPABASE_URL')
    # Use service role key if available for administrative privileges, otherwise anon key
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("✗ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set in .env")
        sys.exit(1)

    if 'service_role' in supabase_key:
        print("✓ Using service_role key (bypasses RLS)")
    else:
        print("⚠ Warning: Not using service_role key. Deletions might fail if RLS policies prevent them.")

    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Tables to clear in order (Child -> Parent to avoid FK constraints)
    tables_to_clear = ['Submission', 'Team', 'User']
    
    print("⚠ WARNING: This will delete ALL data from the following tables:")
    for table in tables_to_clear:
        print(f"  - {table}")
    
    confirm = input("\nType 'yes' to confirm deletion: ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return

    for table in tables_to_clear:
        try:
            print(f"Clearing table '{table}'...")
            
            # Determine primary key column based on table name
            if table == 'User':
                # User table uses 'email' as primary key
                response = supabase.table(table).delete().neq('email', 'placeholder_impossible_email').execute()
            else:
                # Other tables use 'id'
                response = supabase.table(table).delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            
            print(f"✓ Cleared table '{table}'")
            
        except Exception as e:
            print(f"✗ Error clearing table '{table}': {e}")
            print("  (Make sure the table exists and the primary key column exists. If using RLS, ensure you have delete policy.)")
            choice = input("Continue to next table? (y/n): ")
            if choice.lower() != 'y':
                sys.exit(1)

    print("\n✓ Database clearing complete.")

if __name__ == "__main__":
    clear_tables()
