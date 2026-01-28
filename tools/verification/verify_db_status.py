
import os
import sys
from database.manager import get_db

def main():
    print("Verifying Database Status...")
    db = get_db()
    health = db.health_check()
    print(f"Health Check: {health}")

    if health.get('database_exists') and health.get('schema_version'):
        print("SUCCESS: Database is healthy and schema is valid.")
        sys.exit(0)
    else:
        print("FAILURE: Database check failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
