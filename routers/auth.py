from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from db import SessionLocal
from schemas.auth_schemas import LoginOTPRequest, SubmitOTPRequest, AuthValidationRequest
from services.auth_service import (
    login_with_email_generate_otp,
    login_submit_otp,
    auth_validation_by_token_and_guest_account_id
)

router = APIRouter(prefix="/auth", tags=["authentication"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/email-get-otp")
def login_with_email_generate_otp_endpoint(payload: LoginOTPRequest, db: Session = Depends(get_db)):
    return login_with_email_generate_otp(payload.email, payload.is_from_login, db)

@router.post("/submit-otp")
def login_submit_otp_endpoint(payload: SubmitOTPRequest, db: Session = Depends(get_db)):
    return login_submit_otp(payload.email, payload.otp, db)

@router.post("/validate-by-token-and-guest-account-id")
def auth_validation_endpoint(request: Request, payload: AuthValidationRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return auth_validation_by_token_and_guest_account_id(payload.guest_account_id, guest_account_token, db)
