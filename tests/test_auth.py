from datetime import timedelta

import pytest
from argon2 import PasswordHasher

from api import create_app
from extensions import db
from models import EmailVerificationCode, User, utcnow


PASSWORD = "ClaveSegura123"
REQUIRED_ENV = (
    "DATABASE_URL", "JWT_SECRET_KEY", "VERIFICATION_CODE_SECRET", "MAIL_SERVER", "MAIL_PORT",
    "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_DEFAULT_SENDER",
)


@pytest.fixture()
def auth_app(tmp_path):
    app = create_app({
        "TESTING": True,
        "AUTH_DISABLED": False,
        "AUTH_CONFIG_MISSING": [],
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'auth.db'}",
        "JWT_SECRET_KEY": "test-jwt-secret-not-for-production",
        "VERIFICATION_CODE_SECRET": "test-code-secret-not-for-production",
        "JWT_COOKIE_SECURE": False,
    })
    with app.app_context():
        db.create_all()
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def auth_client(auth_app):
    return auth_app.test_client()


def register(client, email="alumno@uadec.edu.mx"):
    return client.post("/api/auth/register", json={
        "full_name": "Alumno UAdeC", "email": email, "password": PASSWORD,
    })


def register_and_verify(app, client, email="alumno@uadec.edu.mx"):
    assert register(client, email).status_code == 201
    code = app.extensions["mailer"].outbox[-1]["code"]
    assert client.post("/api/auth/verify", json={"email": email, "code": code}).status_code == 200


def csrf(client, name="csrf_access_token"):
    return client.get_cookie(name).value


def test_production_rejects_missing_auth_configuration(monkeypatch):
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FLASK_ENV", "production")
    with pytest.raises(RuntimeError, match="Falta configuración requerida"):
        create_app()


def test_registration_validates_domain_password_duplicate_and_hashes(auth_app, auth_client):
    assert register(auth_client, "user@other.edu.mx").status_code == 400
    weak = auth_client.post("/api/auth/register", json={
        "full_name": "User", "email": "user@uadec.edu.mx", "password": "weakpassword",
    })
    assert weak.status_code == 400
    assert register(auth_client, "ALUMNO@UADEC.EDU.MX").status_code == 201
    assert register(auth_client).status_code == 409

    with auth_app.app_context():
        user = db.session.execute(db.select(User)).scalar_one()
        record = db.session.execute(db.select(EmailVerificationCode)).scalar_one()
        code = auth_app.extensions["mailer"].outbox[0]["code"]
        assert user.email == "alumno@uadec.edu.mx"
        assert user.password_hash != PASSWORD
        assert PasswordHasher().verify(user.password_hash, PASSWORD)
        assert record.code_hash != code and code not in record.code_hash
        assert record.expires_at > record.created_at


def test_verification_expiration_attempt_limit_and_resend(auth_app, auth_client):
    assert register(auth_client).status_code == 201
    first_code = auth_app.extensions["mailer"].outbox[-1]["code"]
    assert auth_client.post("/api/auth/resend-code", json={"email": "alumno@uadec.edu.mx"}).status_code == 429

    with auth_app.app_context():
        record = db.session.execute(db.select(EmailVerificationCode)).scalar_one()
        record.last_sent_at = utcnow() - timedelta(seconds=61)
        db.session.commit()
    assert auth_client.post("/api/auth/resend-code", json={"email": "alumno@uadec.edu.mx"}).status_code == 200
    assert len(auth_app.extensions["mailer"].outbox) == 2
    second_code = auth_app.extensions["mailer"].outbox[-1]["code"]
    assert first_code != second_code
    assert auth_client.post("/api/auth/verify", json={"email": "alumno@uadec.edu.mx", "code": first_code}).status_code == 400

    wrong_code = "000001" if second_code == "000000" else "000000"
    for _ in range(5):
        response = auth_client.post("/api/auth/verify", json={"email": "alumno@uadec.edu.mx", "code": wrong_code})
        assert response.status_code == 400
    assert auth_client.post("/api/auth/verify", json={"email": "alumno@uadec.edu.mx", "code": second_code}).status_code == 400

    with auth_app.app_context():
        latest = db.session.execute(
            db.select(EmailVerificationCode).order_by(EmailVerificationCode.id.desc())
        ).scalars().first()
        latest.consumed_at = None
        latest.attempt_count = 0
        latest.expires_at = utcnow() - timedelta(seconds=1)
        db.session.commit()
    assert auth_client.post("/api/auth/verify", json={"email": "alumno@uadec.edu.mx", "code": second_code}).status_code == 400


def test_login_cookies_me_refresh_logout_and_api_protection(auth_app, auth_client):
    assert auth_client.post("/api/data/manual", json={}).status_code == 401
    assert register(auth_client).status_code == 201
    before = auth_client.post("/api/auth/login", json={"email": "alumno@uadec.edu.mx", "password": PASSWORD})
    assert before.status_code == 401
    code = auth_app.extensions["mailer"].outbox[-1]["code"]
    assert auth_client.post("/api/auth/verify", json={"email": "alumno@uadec.edu.mx", "code": code}).status_code == 200

    login = auth_client.post("/api/auth/login", json={"email": "ALUMNO@UADEC.EDU.MX", "password": PASSWORD})
    assert login.status_code == 200
    cookies = login.headers.getlist("Set-Cookie")
    assert any("access_token_cookie=" in value and "HttpOnly" in value and "SameSite=Lax" in value for value in cookies)
    assert any("refresh_token_cookie=" in value and "HttpOnly" in value for value in cookies)
    assert auth_client.get("/api/auth/me").status_code == 200

    created = auth_client.post(
        "/api/data/manual",
        json={"name": "protected", "columns": ["x"], "rows": [[1]]},
        headers={"X-CSRF-TOKEN": csrf(auth_client)},
    )
    assert created.status_code == 201
    refreshed = auth_client.post(
        "/api/auth/refresh", headers={"X-CSRF-TOKEN": csrf(auth_client, "csrf_refresh_token")}
    )
    assert refreshed.status_code == 200
    logout = auth_client.post("/api/auth/logout", headers={"X-CSRF-TOKEN": csrf(auth_client)})
    assert logout.status_code == 200
    assert auth_client.get("/api/auth/me").status_code == 401


def test_login_error_does_not_enumerate_accounts(auth_client):
    missing = auth_client.post("/api/auth/login", json={"email": "missing@uadec.edu.mx", "password": PASSWORD})
    assert register(auth_client).status_code == 201
    existing = auth_client.post("/api/auth/login", json={"email": "alumno@uadec.edu.mx", "password": PASSWORD})
    assert missing.status_code == existing.status_code == 401
    assert missing.get_json() == existing.get_json()


def test_registration_rolls_back_when_email_delivery_fails(auth_app, auth_client):
    class FailingMailer:
        def send_verification(self, _recipient, _code):
            raise OSError("SMTP no disponible")

    auth_app.extensions["mailer"] = FailingMailer()
    response = register(auth_client)
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "email_delivery_failed"
    with auth_app.app_context():
        assert db.session.scalar(db.select(db.func.count(User.id))) == 0
