from sqlalchemy.orm import Session
from datetime import datetime, timezone
from models import ServicePrice, Currency, Country, Discount
from enums import DiscountType, DiscountCategory

def list_service_prices(db: Session):
    """List all service prices"""
    now = datetime.now(timezone.utc)

    service_rows = (
        db.query(ServicePrice, Currency)
        .join(Currency, ServicePrice.currency == Currency.currency_id)
        .filter(ServicePrice.is_show == True)
        .order_by(ServicePrice.sort_order.asc())
        .all()
    )

    from sqlalchemy import or_
    discount_rows = (
        db.query(Discount)
        .filter(
            Discount.start_date <= now,
            Discount.end_date >= now,
            or_(
                Discount.discount_category == int(DiscountCategory.General),
                Discount.discount_category == int(DiscountCategory.Referral_Code)
            )
        )
        .order_by(Discount.discount_category.asc(), Discount.start_date.asc())
        .all()
    )

    def map_discount(disc):
        return {
            "discount_id": str(disc.discount_id),
            "discount_name": disc.discount_name,
            "discount_description": disc.discount_description,
            "discount_type": disc.discount_type,
            "discount_type_name": DiscountType(disc.discount_type).name.replace("_", " ") if disc.discount_type else None,
            "discount_category": disc.discount_category,
            "discount_category_name": DiscountCategory(disc.discount_category).name.replace("_", " ") if disc.discount_category else None,
            "reference_id": str(disc.reference_id),
            "discount_value": float(disc.discount_value),
            "start_date": disc.start_date.isoformat(),
            "end_date": disc.end_date.isoformat(),
        }

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
            "discounts": [
                map_discount(d) for d in discount_rows
                if d.reference_id == sp.service_price_id
            ],
        }
        for sp, cur in service_rows
    ]

def get_service_price(service_price_id: str, db: Session):
    """Get service price by ID"""
    now = datetime.now(timezone.utc)
    from sqlalchemy import or_
    q = (
        db.query(ServicePrice, Currency)
        .join(Currency, ServicePrice.currency == Currency.currency_id)
        .filter(ServicePrice.service_price_id == service_price_id)
    )
    row = q.first()
    if not row:
        return {"error": "not_found"}
    sp, cur = row

    discount_rows = (
        db.query(Discount)
        .filter(
            Discount.reference_id == sp.service_price_id,
            Discount.start_date <= now,
            Discount.end_date >= now,
            or_(
                Discount.discount_category == int(DiscountCategory.General),
                Discount.discount_category == int(DiscountCategory.Referral_Code)
            )
        )
        .order_by(Discount.discount_category.asc(), Discount.start_date.asc())
        .all()
    )

    return {
        "service_price_id": str(sp.service_price_id),
        "service_name": sp.service_name,
        "price": float(sp.price) if sp.price is not None else None,
        "currency": str(sp.currency),
        "currency_code": cur.currency_code,
        "currency_symbol": cur.currency_symbol,
        "discounts": [
            {
                "discount_id": str(d.discount_id),
                "discount_name": d.discount_name,
                "discount_description": d.discount_description,
                "discount_type": d.discount_type,
                "discount_type_name": DiscountType(d.discount_type).name.replace("_", " ") if d.discount_type else None,
                "discount_category": d.discount_category,
                "discount_category_name": DiscountCategory(d.discount_category).name.replace("_", " ") if d.discount_category else None,
                "reference_id": str(d.reference_id),
                "discount_value": float(d.discount_value),
                "start_date": d.start_date.isoformat(),
                "end_date": d.end_date.isoformat(),
            }
            for d in discount_rows
        ],
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
