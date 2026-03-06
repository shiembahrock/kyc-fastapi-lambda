from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import SessionLocal
from schemas.order_schemas import CheckoutStartRequest
from services.checkout_service import checkout_start

router = APIRouter(prefix="/checkout", tags=["checkout"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/start")
def checkout_start_endpoint(payload: CheckoutStartRequest, db: Session = Depends(get_db)):
    return checkout_start(
        payload.email,
        payload.first_name,
        payload.last_name,
        payload.company_name,
        payload.country_id,
        payload.service_id,
        payload.price,
        payload.currency_id,
        payload.currency_code,
        payload.cancel_url,
        payload.success_url,
        db
    )
