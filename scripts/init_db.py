import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import create_app
from extensions import db


app = create_app()

with app.app_context():
    db.create_all()
    print("Base de datos inicializada.")
