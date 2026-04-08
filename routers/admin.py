from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from db import SessionLocal
from services.auth_administrator_service import create_administrator, login_administrator

router = APIRouter(prefix="/admin", tags=["admin"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class CreateAdministratorRequest(BaseModel):
    email: EmailStr

@router.post("/create-administrator")
def create_administrator_endpoint(payload: CreateAdministratorRequest, db: Session = Depends(get_db)):
    return create_administrator(payload.email, db)

class LoginAdministratorRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/login")
def login_administrator_endpoint(payload: LoginAdministratorRequest, db: Session = Depends(get_db)):
    return login_administrator(payload.email, payload.password, db)
