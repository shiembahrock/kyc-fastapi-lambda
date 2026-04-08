import os
import json
import random
import string
import logging
import bcrypt
from sqlalchemy.orm import Session
from models import AdminUser
from utils.lambda_client import lambda_client

WEBHOOK_TARGET_LAMBDA_ARN = os.getenv("WEBHOOK_TARGET_LAMBDA_ARN", "")
logger = logging.getLogger()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def create_administrator(email: str, db: Session):
    existing = db.query(AdminUser).filter(AdminUser.email == email).first()
    if existing:
        return {"success": False, "message": "Email is already existed."}

    chars = string.ascii_letters + string.digits
    password = (
        random.choice(string.ascii_uppercase) +
        random.choice(string.digits) +
        "".join(random.choices(chars, k=6))
    )
    password = "".join(random.sample(password, len(password)))

    try:
        admin = AdminUser(email=email, password=hash_password(password))
        db.add(admin)
        db.commit()
    except Exception:
        db.rollback()
        return {"success": False, "message": "Failed to store data."}

    if WEBHOOK_TARGET_LAMBDA_ARN:
        payload = {
            "action": "send_email_smtp",
            "payload": {
                "to_email": email,
                "subject": "Enigmatig KYC & AML - Administrator Invitation.",
                "body": f"Hi,<br/><br/>Your temporary password is: {password}.<br/>Please use your temporary password into the login box to connect to your account.",
                "is_html": True
            }
        }
        try:
            lambda_client.invoke(
                FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8")
            )
            return {"success": True, "message": "Email sent successfully."}
        except Exception:
            logger.exception("create_administrator: failed to invoke lambda")
            return {"success": False, "message": "Email failed to send."}

    return {"success": True, "message": "Administrator created."}

def login_administrator(email: str, password: str, db: Session):
    admin = db.query(AdminUser).filter(AdminUser.email == email).first()
    if not admin or not admin.password:
        return {"success": False, "message": "Invalid email or password."}
    if not bcrypt.checkpw(password.encode("utf-8"), admin.password.encode("utf-8")):
        return {"success": False, "message": "Invalid email or password."}
    if not admin.is_activated:
        return {"success": False, "message": "Account is not activated."}
    return {"success": True, "message": "Login successful."}
