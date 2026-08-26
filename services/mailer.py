import smtplib
from email.message import EmailMessage


class SMTPMailer:
    def __init__(self, app):
        self.config = app.config

    def send_verification(self, recipient: str, code: str) -> None:
        message = EmailMessage()
        message["Subject"] = "Verifica tu cuenta de LoboStats"
        message["From"] = self.config["MAIL_DEFAULT_SENDER"]
        message["To"] = recipient
        message.set_content(
            f"Tu código de verificación de LoboStats es {code}. "
            "Expira en 15 minutos. Si no solicitaste esta cuenta, ignora este correo."
        )
        message.add_alternative(
            "<html><body><h2>Verifica tu cuenta de LoboStats</h2>"
            f"<p>Tu código de verificación es <strong>{code}</strong>.</p>"
            "<p>Expira en 15 minutos. Si no solicitaste esta cuenta, ignora este correo.</p>"
            "</body></html>",
            subtype="html",
        )
        with smtplib.SMTP(self.config["MAIL_SERVER"], self.config["MAIL_PORT"], timeout=10) as smtp:
            if self.config["MAIL_USE_TLS"]:
                smtp.starttls()
            if self.config.get("MAIL_USERNAME"):
                smtp.login(self.config["MAIL_USERNAME"], self.config["MAIL_PASSWORD"])
            smtp.send_message(message)


class CapturingMailer:
    def __init__(self):
        self.outbox = []

    def send_verification(self, recipient: str, code: str) -> None:
        self.outbox.append({"to": recipient, "code": code})
