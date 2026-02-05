#!/usr/bin/env python3
"""
Script to seed a Supabase table with data from a CSV file.
"""

import csv
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()


def seed_table(csv_file_path: str, supabase_url: str, supabase_key: str, table_name: str):
    """
    Seed a Supabase table with data from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        supabase_url: Supabase project URL
        supabase_key: Supabase anon/service key
        table_name: Name of the table to seed
    """
    # Debug: Check key type
    if 'service_role' in supabase_key:
        print("✓ Using service_role key (bypasses RLS)")
    else:
        print("⚠ Warning: Not using service_role key - may face RLS restrictions")
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # CSV columns: Reg. No., Name, Email
    # Table columns: email, name, regNo
    records = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Map CSV columns to table columns
                record = {
                    'regNo': row['Reg. No.'],
                    'email': row['Email'],
                    'name': row['Name']
                }
                records.append(record)
        
        print(f"✓ Read {len(records)} records from {csv_file_path}")
        
    except FileNotFoundError:
        print(f"✗ Error: CSV file not found at {csv_file_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"✗ Error: Missing required column {e} in CSV file")
        sys.exit(1)
    
    # Deduplicate records by email (keep last occurrence)
    unique_records = {}
    for record in records:
        unique_records[record['email']] = record
    
    deduplicated_records = list(unique_records.values())
    
    if len(deduplicated_records) < len(records):
        duplicates_count = len(records) - len(deduplicated_records)
        print(f"⚠ Found {duplicates_count} duplicate email(s), keeping only unique records")
        print(f"✓ {len(deduplicated_records)} unique records to process")
    
    # Insert records into Supabase
    if not deduplicated_records:
        print("⚠ No records to insert")
        return
    
    try:
        # Upsert in batches to avoid payload size limits (insert or update on conflict)
        batch_size = 100
        total_processed = 0
        
        for i in range(0, len(deduplicated_records), batch_size):
            batch = deduplicated_records[i:i + batch_size]
            # Use upsert to handle duplicates (updates existing, inserts new)
            response = supabase.table(table_name).upsert(batch).execute()
            total_processed += len(batch)
            print(f"✓ Processed batch {i // batch_size + 1}: {len(batch)} records (Total: {total_processed}/{len(deduplicated_records)})")
        
        print(f"\n✓ Successfully seeded {total_processed} records into '{table_name}' table (inserted new + updated existing)")
        
    except Exception as e:
        print(f"✗ Error inserting data into Supabase: {e}")
        sys.exit(1)


def main():
    """Main function to run the seeding script."""
    # Get configuration from environment variables or command-line arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else input("Enter CSV file path: ")
    
    supabase_url = os.getenv('SUPABASE_URL') or input("Enter Supabase URL: ")
    supabase_key = os.getenv('SUPABASE_ANON_KEY') or input("Enter Supabase Anon Key: ")
    table_name = os.getenv('SUPABASE_TABLE_NAME') or input("Enter table name: ")
    
    # Validate inputs
    if not all([csv_file, supabase_url, supabase_key, table_name]):
        print("✗ Error: All parameters are required")
        sys.exit(1)
    
    print(f"\n📊 Seeding Configuration:")
    print(f"   CSV File: {csv_file}")
    print(f"   Table: {table_name}")
    print(f"   Supabase URL: {supabase_url[:30]}...")
    print()
    
    # Run seeding
    seed_table(csv_file, supabase_url, supabase_key, table_name)


if __name__ == "__main__":
    main()
