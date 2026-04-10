from pydantic import BaseModel, EmailStr
from typing import Optional

class OrderCreate(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    company_name: str
    country: str
    product_name: str
    price: Optional[str] = None
    period: Optional[str] = None
    description: Optional[str] = None

class CheckoutStartRequest(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    company_name: str
    country_id: str
    phone: str
    service_id: str
    price: float
    currency_id: str
    currency_code: str
    cancel_url: str
    success_url: str
