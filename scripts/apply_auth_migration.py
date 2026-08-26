import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine


ROOT = Path(__file__).resolve().parent.parent
environment_file = Path(os.getenv("MIGRATION_ENV_FILE", ROOT / ".env"))
load_dotenv(environment_file, override=True)

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise SystemExit("DATABASE_URL es obligatorio.")
if database_url.startswith("mysql://"):
    database_url = "mysql+pymysql://" + database_url[len("mysql://"):]

statements = [
    statement.strip()
    for statement in (ROOT / "migrations" / "001_auth.sql").read_text(encoding="utf-8").split(";")
    if statement.strip()
]

engine = create_engine(database_url, pool_pre_ping=True)
with engine.begin() as connection:
    for statement in statements:
        connection.exec_driver_sql(statement)

print(f"Migración de autenticación aplicada: {len(statements)} sentencias.")
