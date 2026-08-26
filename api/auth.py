import hashlib
import hmac
import re
import secrets
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_csrf_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
    set_access_cookies,
    set_refresh_cookies,
    unset_jwt_cookies,
)
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from extensions import db
from models import EmailVerificationCode, User, utcnow
from utils.errors import APIError


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
password_hasher = PasswordHasher()
DUMMY_PASSWORD_HASH = password_hasher.hash("DummyPassword123")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@uadec\.edu\.mx$", re.IGNORECASE)


@auth_bp.before_request
def check_auth_config():
    require_auth_config()


def normalize_email(value) -> str:
    return str(value or "").strip().lower()


def validate_password(password) -> None:
    if not isinstance(password, str) or len(password) < 12 or not re.search(r"[A-Z]", password) \
            or not re.search(r"[a-z]", password) or not re.search(r"\d", password):
        raise APIError("La contraseña debe tener al menos 12 caracteres, mayúscula, minúscula y número.")


def require_auth_config() -> None:
    missing = current_app.config.get("AUTH_CONFIG_MISSING", [])
    if missing:
        raise APIError("La autenticación no está configurada.", 503, "auth_not_configured")


def code_digest(user_id: int, code: str) -> str:
    secret = current_app.config["VERIFICATION_CODE_SECRET"].encode()
    return hmac.new(secret, f"{user_id}:{code}".encode(), hashlib.sha256).hexdigest()


def create_and_send_code(user: User) -> EmailVerificationCode:
    now = utcnow()
    db.session.execute(
        update(EmailVerificationCode)
        .where(EmailVerificationCode.user_id == user.id, EmailVerificationCode.consumed_at.is_(None))
        .values(consumed_at=now)
    )
    while True:
        code = f"{secrets.randbelow(1_000_000):06d}"
        digest = code_digest(user.id, code)
        reused = db.session.scalar(
            select(EmailVerificationCode.id).where(
                EmailVerificationCode.user_id == user.id,
                EmailVerificationCode.code_hash == digest,
            )
        )
        if not reused:
            break
    record = EmailVerificationCode(
        user_id=user.id,
        code_hash=digest,
        expires_at=now + timedelta(minutes=15),
        last_sent_at=now,
    )
    db.session.add(record)
    try:
        current_app.extensions["mailer"].send_verification(user.email, code)
    except Exception as error:
        current_app.logger.exception("No se pudo enviar el código de verificación")
        raise APIError(
            "No se pudo enviar el correo de verificación. Intenta de nuevo más tarde.",
            503,
            "email_delivery_failed",
        ) from error
    return record


def payload() -> dict:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise APIError("Se requiere un cuerpo JSON.")
    return value


@auth_bp.post("/register")
def register():
    require_auth_config()
    data = payload()
    email = normalize_email(data.get("email"))
    full_name = str(data.get("full_name") or "").strip()
    password = data.get("password")
    if not full_name or len(full_name) > 150:
        raise APIError("El nombre completo es obligatorio y no puede exceder 150 caracteres.")
    if not EMAIL_PATTERN.fullmatch(email):
        raise APIError("Usa una dirección del dominio @uadec.edu.mx.")
    validate_password(password)
    user = User(full_name=full_name, email=email, password_hash=password_hasher.hash(password))
    try:
        db.session.add(user)
        db.session.flush()
        create_and_send_code(user)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        raise APIError("La cuenta ya está registrada.", 409, "email_already_registered")
    except Exception:
        db.session.rollback()
        raise
    return jsonify({"message": "Registro creado. Revisa tu correo para verificar la cuenta."}), 201


@auth_bp.post("/verify")
def verify():
    require_auth_config()
    data = payload()
    email = normalize_email(data.get("email"))
    code = str(data.get("code") or "")
    if not re.fullmatch(r"\d{6}", code):
        raise APIError("Código inválido.", 400, "invalid_verification_code")
    user = db.session.scalar(select(User).where(User.email == email))
    if not user:
        raise APIError("Código inválido o expirado.", 400, "invalid_verification_code")
    record = db.session.scalar(
        select(EmailVerificationCode)
        .where(EmailVerificationCode.user_id == user.id, EmailVerificationCode.consumed_at.is_(None))
        .order_by(EmailVerificationCode.created_at.desc())
    )
    now = utcnow()
    if not record or record.expires_at.replace(tzinfo=record.expires_at.tzinfo or now.tzinfo) <= now \
            or record.attempt_count >= record.max_attempts:
        raise APIError("Código inválido o expirado.", 400, "invalid_verification_code")
    if not hmac.compare_digest(record.code_hash, code_digest(user.id, code)):
        record.attempt_count += 1
        if record.attempt_count >= record.max_attempts:
            record.consumed_at = now
        db.session.commit()
        raise APIError("Código inválido o expirado.", 400, "invalid_verification_code")
    record.consumed_at = now
    user.email_verified = True
    db.session.commit()
    return jsonify({"message": "Correo verificado correctamente."})


@auth_bp.post("/resend-code")
def resend_code():
    require_auth_config()
    email = normalize_email(payload().get("email"))
    user = db.session.scalar(select(User).where(User.email == email))
    if not user or user.email_verified:
        return jsonify({"message": "Si la cuenta existe y está pendiente, se enviará un código."})
    latest = db.session.scalar(
        select(EmailVerificationCode).where(EmailVerificationCode.user_id == user.id)
        .order_by(EmailVerificationCode.last_sent_at.desc())
    )
    now = utcnow()
    if latest:
        sent_at = latest.last_sent_at.replace(tzinfo=latest.last_sent_at.tzinfo or now.tzinfo)
        if now - sent_at < timedelta(seconds=60):
            raise APIError("Espera 60 segundos antes de solicitar otro código.", 429, "resend_too_soon")
    create_and_send_code(user)
    db.session.commit()
    return jsonify({"message": "Si la cuenta existe y está pendiente, se enviará un código."})


@auth_bp.post("/login")
def login():
    require_auth_config()
    data = payload()
    user = db.session.scalar(select(User).where(User.email == normalize_email(data.get("email"))))
    try:
        valid_password = password_hasher.verify(
            user.password_hash if user else DUMMY_PASSWORD_HASH,
            str(data.get("password") or ""),
        ) and user is not None
    except VerifyMismatchError:
        valid_password = False
    if not valid_password or not user.email_verified or not user.is_active:
        raise APIError("Credenciales inválidas.", 401, "invalid_credentials")
    if password_hasher.check_needs_rehash(user.password_hash):
        user.password_hash = password_hasher.hash(data["password"])
    user.last_login_at = utcnow()
    db.session.commit()
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    response = jsonify({
        "user": user.to_dict(),
        "csrf_access_token": get_csrf_token(access_token),
        "csrf_refresh_token": get_csrf_token(refresh_token),
    })
    set_access_cookies(response, access_token)
    set_refresh_cookies(response, refresh_token)
    return response


@auth_bp.post("/refresh")
@jwt_required(refresh=True)
def refresh():
    access_token = create_access_token(identity=get_jwt_identity())
    response = jsonify({"message": "Sesión renovada.", "csrf_access_token": get_csrf_token(access_token)})
    set_access_cookies(response, access_token)
    return response


@auth_bp.get("/refresh-csrf")
@jwt_required(refresh=True)
def refresh_csrf():
    return jsonify({"csrf_refresh_token": get_jwt()["csrf"]})


@auth_bp.post("/logout")
def logout():
    response = jsonify({"message": "Sesión cerrada."})
    unset_jwt_cookies(response)
    return response


@auth_bp.get("/me")
@jwt_required()
def me():
    user = db.session.get(User, int(get_jwt_identity()))
    if not user or not user.is_active:
        raise APIError("Sesión inválida.", 401, "invalid_session")
    return jsonify({"user": user.to_dict(), "csrf_access_token": get_jwt()["csrf"]})
