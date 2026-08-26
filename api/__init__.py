from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from api.charts import charts_bp
from api.anova import anova_bp
from api.data import data_bp
from api.descriptive import descriptive_bp
from api.distributions import distributions_bp
from api.hypothesis import hypothesis_bp
from api.statistics import (basic_stats_bp, correlation_bp, nonparametric_bp, power_bp,
                            proportion_bp, variance_bp)
from modules.dataset_store import DatasetStore
from utils.errors import register_error_handlers


def create_app(config: dict | None = None) -> Flask:
    """Crea y configura la aplicación Flask."""
    root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, static_folder=str(root / "static"))
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        JSON_SORT_KEYS=False,
        TESTING=False,
    )
    if config:
        app.config.update(config)

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.extensions["dataset_store"] = DatasetStore()
    register_error_handlers(app)

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
