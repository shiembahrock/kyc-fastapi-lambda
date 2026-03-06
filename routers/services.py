from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from db import SessionLocal
from services.service_service import (
    list_service_prices,
    get_service_price,
    list_countries
)
from services.guest_service import get_service_info_by_order_code

router = APIRouter(tags=["services"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/service-prices")
def list_service_prices_endpoint(db: Session = Depends(get_db)):
    return list_service_prices(db)

@router.get("/service-prices/{service_price_id}")
def get_service_price_endpoint(service_price_id: str, db: Session = Depends(get_db)):
    return get_service_price(service_price_id, db)

@router.get("/countries")
def list_countries_endpoint(db: Session = Depends(get_db)):
    return list_countries(db)

@router.get("/service-info/{order_code}")
def get_service_info_by_order_code_endpoint(order_code: str, request: Request, db: Session = Depends(get_db)):
    guest_account_id = request.headers.get("GuestAccountId", "")
    guest_login_token = request.headers.get("GuestLoginToken", "")
    return get_service_info_by_order_code(order_code, guest_account_id, guest_login_token, db)
