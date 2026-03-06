from pydantic import BaseModel, EmailStr

class LoginOTPRequest(BaseModel):
    email: EmailStr
    is_from_login: bool = False

class SubmitOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class AuthValidationRequest(BaseModel):
    guest_account_id: str
