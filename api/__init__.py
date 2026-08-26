import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import verify_jwt_in_request
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

from api.auth import auth_bp
from api.charts import charts_bp
from api.anova import anova_bp
from api.data import data_bp
from api.descriptive import descriptive_bp
from api.distributions import distributions_bp
from api.hypothesis import hypothesis_bp
from api.statistics import (basic_stats_bp, correlation_bp, nonparametric_bp, power_bp,
                            proportion_bp, variance_bp)
from modules.dataset_store import DatasetStore
from extensions import db, jwt
from services.mailer import CapturingMailer, SMTPMailer
from utils.errors import register_error_handlers


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def create_app(config: dict | None = None) -> Flask:
    """Crea y configura la aplicación Flask."""
    app = Flask(__name__, static_folder=str(ROOT / "static"))
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("mysql://"):
        database_url = "mysql+pymysql://" + database_url[len("mysql://"):]
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        JSON_SORT_KEYS=False,
        TESTING=False,
        AUTH_DISABLED=_env_bool("AUTH_DISABLED"),
        SQLALCHEMY_DATABASE_URI=database_url or "sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY") or secrets.token_urlsafe(48),
        VERIFICATION_CODE_SECRET=os.getenv("VERIFICATION_CODE_SECRET"),
        JWT_TOKEN_LOCATION=["cookies"],
        JWT_COOKIE_CSRF_PROTECT=True,
        JWT_CSRF_IN_COOKIES=False,
        JWT_COOKIE_SAMESITE="Lax",
        JWT_COOKIE_SECURE=_env_bool("JWT_COOKIE_SECURE"),
        MAIL_SERVER=os.getenv("MAIL_SERVER"),
        MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
        MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
        MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
        MAIL_USE_TLS=_env_bool("MAIL_USE_TLS", True),
        MAIL_DEFAULT_SENDER=os.getenv("MAIL_DEFAULT_SENDER"),
    )
    if config:
        app.config.update(config)

    missing = [name for name in (
        "DATABASE_URL", "JWT_SECRET_KEY", "VERIFICATION_CODE_SECRET", "MAIL_SERVER",
        "MAIL_PORT", "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_DEFAULT_SENDER",
    ) if not os.getenv(name)]
    if app.config["TESTING"]:
        missing = app.config.get("AUTH_CONFIG_MISSING", [])
    app.config["AUTH_CONFIG_MISSING"] = missing
    if os.getenv("FLASK_ENV", "").lower() == "production" and missing:
        raise RuntimeError(f"Falta configuración requerida: {', '.join(missing)}")

    origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError("CORS_ORIGINS no puede contener '*' cuando credentials está activado.")
    CORS(app, resources={r"/api/*": {"origins": origins}}, supports_credentials=True)
    db.init_app(app)
    jwt.init_app(app)
    app.extensions["dataset_store"] = DatasetStore()
    app.extensions["mailer"] = app.config.get("MAILER") or (CapturingMailer() if app.config["TESTING"] else SMTPMailer(app))
    register_error_handlers(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(descriptive_bp)
    app.register_blueprint(charts_bp)
    app.register_blueprint(hypothesis_bp)
    app.register_blueprint(basic_stats_bp)
    app.register_blueprint(variance_bp)
    app.register_blueprint(nonparametric_bp)
    app.register_blueprint(proportion_bp)
    app.register_blueprint(correlation_bp)
    app.register_blueprint(power_bp)
    app.register_blueprint(anova_bp)
    app.register_blueprint(distributions_bp)

    public_paths = {"/api/health", "/api/docs", "/api/openapi.yaml"}

    @app.before_request
    def protect_api():
        if request.method == "OPTIONS" or app.config["AUTH_DISABLED"] or not request.path.startswith("/api/"):
            return None
        if request.path.startswith("/api/auth/") or request.path in public_paths:
            return None
        verify_jwt_in_request()

    @app.cli.command("init-db")
    def init_db_command():
        """Crea las tablas configuradas que todavía no existan."""
        db.create_all()
        print("Base de datos inicializada.")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "LoboStats", "version": "1.0.0"})

    @app.get("/api/openapi.yaml")
    def openapi():
        return send_from_directory(app.static_folder, "openapi.yaml", mimetype="application/yaml")

    @app.get("/api/docs")
    def docs():
        return send_from_directory(app.static_folder, "docs.html")

    return app
