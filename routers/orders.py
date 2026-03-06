from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import SessionLocal
from schemas.order_schemas import OrderCreate
from services.order_service import (
    create_order,
    list_orders,
    get_order,
    get_order_payment_by_code
)

router = APIRouter(tags=["orders"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/orders")
def create_order_endpoint(payload: OrderCreate, db: Session = Depends(get_db)):
    return create_order(
        payload.email,
        payload.first_name,
        payload.last_name,
        payload.company_name,
        payload.country,
        payload.product_name,
        payload.price,
        payload.period,
        payload.description,
        db
    )

@router.get("/orders")
def list_orders_endpoint(db: Session = Depends(get_db)):
    return list_orders(db)

@router.get("/orders/{order_id}")
def get_order_endpoint(order_id: str, db: Session = Depends(get_db)):
    return get_order(order_id, db)

@router.get("/order-payments/by-code/{order_code}")
def get_order_payment_by_code_endpoint(order_code: str, db: Session = Depends(get_db)):
    return get_order_payment_by_code(order_code, db)
