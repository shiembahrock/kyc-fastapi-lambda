from sqlalchemy.orm import Session
from models import Order, OrderPayment, Currency

def create_order(email: str, first_name: str, last_name: str, company_name: str, country: str, product_name: str, price: str, period: str, description: str, db: Session):
    """Create new order"""
    obj = Order(
        email=email,
        first_name=first_name,
        last_name=last_name,
        company_name=company_name,
        country=country,
        product_name=product_name,
        price=price,
        period=period,
        description=description,
    )
    db.add(obj)
    db.commit()
    return {"order_id": str(obj.order_id), "status": "created"}

def list_orders(db: Session):
    """List all orders"""
    rows = db.query(Order).order_by(Order.created_at.desc()).all()
    return [
        {
            "order_id": str(r.order_id),
            "email": r.email,
            "first_name": r.first_name,
            "last_name": r.last_name,
            "company_name": r.company_name,
            "country": r.country,
            "product_name": r.product_name,
            "price": r.price,
            "period": r.period,
            "description": r.description,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]

def get_order(order_id: str, db: Session):
    """Get order by ID"""
    r = db.query(Order).filter(Order.order_id == order_id).first()
    if not r:
        return {"error": "not_found"}
    return {
        "order_id": str(r.order_id),
        "email": r.email,
        "first_name": r.first_name,
        "last_name": r.last_name,
        "company_name": r.company_name,
        "country": r.country,
        "product_name": r.product_name,
        "price": r.price,
        "period": r.period,
        "description": r.description,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }

def get_order_payment_by_code(order_code: str, db: Session):
    """Get order payment by order code"""
    q = (
        db.query(OrderPayment, Currency)
        .outerjoin(Currency, OrderPayment.currency_id == Currency.currency_id)
        .filter(OrderPayment.order_code == order_code)
    )
    row = q.first()
    if not row:
        return {"error": "not_found"}
    op, cur = row
    return {
        "order_payment_id": str(op.order_payment_id),
        "order_code": op.order_code,
        "payment_status": op.payment_status,
        "checkout_session_status": op.checkout_session_status,
        "payment_amount": float(op.payment_amount) if op.payment_amount is not None else None,
        "currency_id": str(op.currency_id) if op.currency_id else None,
        "currency_code": (cur.currency_code if cur else None),
        "currency_symbol": (cur.currency_symbol if cur else None),
        "psp_ref_id": op.psp_ref_id,
        "psp_stripe_payment_intent": op.psp_stripe_payment_intent,
        "psp_stripe_receipt_url": op.psp_stripe_receipt_url,
    }
