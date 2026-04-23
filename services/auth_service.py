import random
import jwt
import os
import logging
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse

from models import (
    GuestAccount,
    GuestAccountOTP,
    GuestLoginSession,
    GuestAccountNotificationSetting,
    GuestAccountReferral,
)
from utils.lambda_client import lambda_client
from services.guest_account_credit_service import insert_guest_account_credit_transaction
from decimal import Decimal
import json as _json

JWT_SECRET = os.getenv("JWT_SECRET", "")
WEBHOOK_TARGET_LAMBDA_ARN = os.getenv("WEBHOOK_TARGET_LAMBDA_ARN", "KYCFastAPIFunctionExternal")
LOGIN_EXPIRY_MINUTES = int(os.getenv("LOGIN_EXPIRY_MINUTES", "1440"))

logger = logging.getLogger()

def gen_login_otp():
    """Generate 4-digit OTP"""
    return str(random.randint(1000, 9999))

def gen_login_token(email: str):
    """Generate JWT login token"""
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "email": email,
            "iat": int(now.timestamp()),
            "jti": str(uuid4())
        },
        JWT_SECRET,
        algorithm="HS256"
    )
    return token

def login_with_email_generate_otp(email: str, is_from_login: bool, db: Session):
    """Generate and send OTP for email login"""
    is_send_email = False
    otp = gen_login_otp()
    
    ga = db.query(GuestAccount).filter(GuestAccount.email == email).first()
    
    if ga:
        gaotp = db.query(GuestAccountOTP).filter(
            GuestAccountOTP.guest_account_id == ga.guest_account_id
        ).first()
        
        if gaotp:
            now = datetime.now(timezone.utc)
            if gaotp.expiry_date <= now or is_from_login:
                gaotp.requested_date = datetime.now(timezone.utc)
                gaotp.otp = otp
                gaotp.expiry_date = datetime.now(timezone.utc) + timedelta(minutes=5)
                db.commit()
                is_send_email = True
        else:
            now = datetime.now(timezone.utc)
            gaotp = GuestAccountOTP(
                guest_account_id=ga.guest_account_id,
                requested_date=now,
                otp=otp,
                expiry_date=now + timedelta(minutes=5)
            )
            db.add(gaotp)
            db.commit()
            is_send_email = True
    else:
        ga = GuestAccount(email=email)
        db.add(ga)
        db.flush()
        
        gans = GuestAccountNotificationSetting(guest_account_id=ga.guest_account_id)
        db.add(gans)
        
        now = datetime.now(timezone.utc)
        gaotp = GuestAccountOTP(
            guest_account_id=ga.guest_account_id,
            requested_date=now,
            otp=otp,
            expiry_date=now + timedelta(minutes=5)
        )
        db.add(gaotp)
        db.commit()
        is_send_email = True
    
    if is_send_email:
        if WEBHOOK_TARGET_LAMBDA_ARN:
            payload = {
                "action": "send_email_smtp",
                "payload": {
                    "to_email": ga.email,
                    "subject": f"Enigmatig KYC & AML - {otp} is your personal code.",
                    "body": f"Hi,<br/><br/>Your personal unique code is: {otp}.<br/>Please type your code into the login box to connect to your account.",
                    "is_html": True
                }
            }
            
            try:
                lambda_client.invoke(
                    FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                    InvocationType="RequestResponse",
                    Payload=_json.dumps(payload).encode("utf-8")
                )
            except Exception as e:
                logger.exception("Error sending OTP email")
        else:
            logger.warning("login_with_email_generate_otp: WEBHOOK_TARGET_LAMBDA_ARN not set; skipping OTP email")
    
    return {"guest_account_id": str(ga.guest_account_id)}

def login_with_registered_email_generate_otp(email: str, is_from_login: bool, db: Session):
    """Generate and send OTP for registered email login only"""
    # Check if email is registered in GuestAccount
    ga = db.query(GuestAccount).filter(GuestAccount.email == email).first()
    
    if not ga:
        return JSONResponse(
            status_code=404,
            content={"message": "email unregistered"}
        )
    
    is_send_email = False
    otp = gen_login_otp()
    
    gaotp = db.query(GuestAccountOTP).filter(
        GuestAccountOTP.guest_account_id == ga.guest_account_id
    ).first()
    
    if gaotp:
        now = datetime.now(timezone.utc)
        if gaotp.expiry_date <= now or is_from_login:
            gaotp.requested_date = datetime.now(timezone.utc)
            gaotp.otp = otp
            gaotp.expiry_date = datetime.now(timezone.utc) + timedelta(minutes=5)
            db.commit()
            is_send_email = True
    else:
        now = datetime.now(timezone.utc)
        gaotp = GuestAccountOTP(
            guest_account_id=ga.guest_account_id,
            requested_date=now,
            otp=otp,
            expiry_date=now + timedelta(minutes=5)
        )
        db.add(gaotp)
        db.commit()
        is_send_email = True
    
    if is_send_email:
        if WEBHOOK_TARGET_LAMBDA_ARN:
            payload = {
                "action": "send_email_smtp",
                "payload": {
                    "to_email": ga.email,
                    "subject": f"Enigmatig KYC & AML - {otp} is your personal code.",
                    "body": f"Hi,<br/><br/>Your personal unique code is: {otp}.<br/>Please type your code into the login box to connect to your account.",
                    "is_html": True
                }
            }
            
            try:
                lambda_client.invoke(
                    FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                    InvocationType="RequestResponse",
                    Payload=_json.dumps(payload).encode("utf-8")
                )
            except Exception as e:
                logger.exception("Error sending OTP email")
        else:
            logger.warning("login_with_registered_email_generate_otp: WEBHOOK_TARGET_LAMBDA_ARN not set; skipping OTP email")
    
    return {"guest_account_id": str(ga.guest_account_id)}

def register_account(
    email: str,
    first_name: str,
    last_name: str,
    company_name: str,
    country_id: str,
    phone: str,
    referral_code: str | None,
    db: Session
):
    """Register guest account and optionally link it to an existing referral code."""
    try:
        existing_guest = db.query(GuestAccount).filter(GuestAccount.email == email).first()
        if existing_guest:
            return JSONResponse(
                status_code=409,
                content={"message": "Email is already registered."}
            )

        referred_by = None
        normalized_referral_code = referral_code.strip() if referral_code else None

        if normalized_referral_code:
            referred_by = db.query(GuestAccountReferral).filter(
                GuestAccountReferral.referral_code == normalized_referral_code
            ).first()

            if not referred_by:
                return JSONResponse(
                    status_code=404,
                    content={"message": "Referral code is not found."}
                )

        guest_account = GuestAccount(
            email=email,
            first_name=first_name,
            last_name=last_name,
            company_name=company_name,
            country_id=country_id,
            phone=phone,
        )
        db.add(guest_account)
        db.flush()

        notification_setting = GuestAccountNotificationSetting(
            guest_account_id=guest_account.guest_account_id
        )
        db.add(notification_setting)

        if referred_by:
            guest_referral = GuestAccountReferral(
                guest_account_id=guest_account.guest_account_id,
                referral_code=None,
                referred_by_id=referred_by.guest_account_referral_id,
            )
            db.add(guest_referral)
            db.flush()
            insert_guest_account_credit_transaction(
                referred_by.guest_account_id,
                guest_referral.guest_account_referral_id,
                1,
                db,
                Decimal("0.00")
            )

        db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "message": "success",
                "guest_account_id": str(guest_account.guest_account_id)
            }
        )
    except Exception:
        db.rollback()
        logger.exception("Error registering guest account")
        return JSONResponse(
            status_code=500,
            content={"message": "failed"}
        )

def login_submit_otp(email: str, otp: str, db: Session):
    """Submit OTP and generate login token"""
    ga = db.query(GuestAccount).filter(GuestAccount.email == email).first()
    
    if not ga:
        return JSONResponse(
            status_code=401,
            content={"message": "The account is not found."}
        )
    
    gaotp = db.query(GuestAccountOTP).filter(
        GuestAccountOTP.guest_account_id == ga.guest_account_id
    ).first()
    
    if not gaotp:
        return JSONResponse(
            status_code=410,
            content={"message": "The code is incorrect. Please check your email."}
        )
    
    if gaotp.otp != otp:
        return JSONResponse(
            status_code=410,
            content={"message": "The code is incorrect. Please check your email."}
        )
    
    now = datetime.now(timezone.utc)
    if gaotp.expiry_date < now:
        return JSONResponse(
            status_code=410,
            content={"message": "OTP has been expired."}
        )
    
    token = gen_login_token(email)
    expiry_on = now + timedelta(minutes=LOGIN_EXPIRY_MINUTES)
    
    gls = db.query(GuestLoginSession).filter(
        GuestLoginSession.guest_account_id == ga.guest_account_id
    ).first()
    
    if gls:
        gls.issued_on = now
        gls.expiry_on = expiry_on
        gls.token = token
    else:
        gls = GuestLoginSession(
            guest_account_id=ga.guest_account_id,
            issued_on=now,
            expiry_on=expiry_on,
            token=token
        )
        db.add(gls)
    
    db.commit()
    
    expiry_timestamp = int(expiry_on.timestamp())
    
    return {
        "message": "Success.",
        "token": token,
        "expiry_on": expiry_timestamp
    }

def auth_validation_by_token_and_guest_account_id(guest_account_id: str, token: str, db: Session):
    """Validate authentication token and extend expiry if valid"""
    result = {
        "auth_status": "",
        "token": token,
        "expiry_on": ""
    }
    
    gls = db.query(GuestLoginSession).filter(
        GuestLoginSession.guest_account_id == guest_account_id,
        GuestLoginSession.token == token
    ).first()
    
    if not gls:
        result["auth_status"] = "not found"
    else:
        now = datetime.now(timezone.utc)
        if gls.expiry_on > now:
            expiry_on = now + timedelta(minutes=LOGIN_EXPIRY_MINUTES)
            gls.expiry_on = expiry_on
            db.commit()
            result["auth_status"] = "valid"
            result["expiry_on"] = int(expiry_on.timestamp())
        else:
            result["auth_status"] = "expired"
            result["expiry_on"] = int(gls.expiry_on.timestamp())
    
    return result
