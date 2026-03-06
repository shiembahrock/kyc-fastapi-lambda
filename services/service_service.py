from sqlalchemy.orm import Session
from models import ServicePrice, Currency, Country

def list_service_prices(db: Session):
    """List all service prices"""
    q = (
        db.query(ServicePrice, Currency)
        .join(Currency, ServicePrice.currency == Currency.currency_id)
        .filter(ServicePrice.is_show == True)
        .order_by(ServicePrice.sort_order.asc())
    )
    rows = q.all()
    return [
        {
            "service_price_id": str(sp.service_price_id),
            "service_name": sp.service_name,
            "price": float(sp.price) if sp.price is not None else None,
            "sort_order": sp.sort_order,
            "is_search_by_credit": sp.is_search_by_credit,
            "search_number": sp.search_number,
            "currency": str(sp.currency),
            "currency_code": cur.currency_code,
            "currency_symbol": cur.currency_symbol,
            "is_popular": sp.is_popular,
        }
        for sp, cur in rows
    ]

def get_service_price(service_price_id: str, db: Session):
    """Get service price by ID"""
    q = (
        db.query(ServicePrice, Currency)
        .join(Currency, ServicePrice.currency == Currency.currency_id)
        .filter(ServicePrice.service_price_id == service_price_id)
    )
    row = q.first()
    if not row:
        return {"error": "not_found"}
    sp, cur = row
    return {
        "service_price_id": str(sp.service_price_id),
        "service_name": sp.service_name,
        "price": float(sp.price) if sp.price is not None else None,
        "currency": str(sp.currency),
        "currency_code": cur.currency_code,
        "currency_symbol": cur.currency_symbol,
    }

def list_countries(db: Session):
    """List all countries"""
    rows = db.query(Country).order_by(Country.country_name.asc()).all()
    return [
        {
            "country_id": str(r.country_id),
            "country_name": r.country_name,
        }
        for r in rows
    ]
