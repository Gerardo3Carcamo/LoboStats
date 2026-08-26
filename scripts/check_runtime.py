import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select


if len(sys.argv) != 2:
    raise SystemExit("Uso: python scripts/check_runtime.py /ruta/app.env")

environment_file = Path(sys.argv[1])
if not environment_file.is_file():
    raise SystemExit(f"No existe el archivo de entorno: {environment_file}")

load_dotenv(environment_file, override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import create_app
from extensions import db
from models import EmailVerificationCode, User


app = create_app()
if app.config.get("AUTH_CONFIG_MISSING"):
    raise SystemExit("La configuración de autenticación está incompleta.")

with app.app_context():
    db.session.execute(select(User.id).limit(1)).all()
    db.session.execute(select(EmailVerificationCode.id).limit(1)).all()
    db.session.remove()

print("Configuración, conexión MySQL y tablas de autenticación verificadas.")
