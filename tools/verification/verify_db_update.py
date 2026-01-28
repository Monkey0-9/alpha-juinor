from database.manager import DatabaseManager

def verify():
    print("Initializing DatabaseManager to trigger schema update...")
    db = DatabaseManager()

    print("Running health check...")
    health = db.health_check()

    print(f"Schema Version: {health['schema_version']}")

    # Check for new tables directly
    conn = db.get_connection()
    new_tables = ["model_decay_metrics", "capital_allocations", "governance_decisions", "strategy_lifecycle"]

    for table in new_tables:
        try:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"✅ Table '{table}' exists.")
        except Exception as e:
            print(f"❌ Table '{table}' MISSING: {e}")

if __name__ == "__main__":
    verify()
