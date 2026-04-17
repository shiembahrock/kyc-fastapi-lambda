from typing import Optional
from pydantic import BaseModel, EmailStr

class LoginOTPRequest(BaseModel):
    email: EmailStr
    is_from_login: bool = False

class SubmitOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class AuthValidationRequest(BaseModel):
    guest_account_id: str

class RegisterAccountRequest(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    company_name: str
    country_id: str
    phone: str
    referral_code: Optional[str] = None
