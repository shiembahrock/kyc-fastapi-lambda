from fastapi import FastAPI, APIRouter, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import uuid4
from pathlib import Path
import json
import random
import jwt
from sqlalchemy.orm import Session
from sqlalchemy import text
from db import Base, engine, SessionLocal
from models import Order, ServicePrice, Currency, Country, OrderPayment, GuestAccount, StripeSetting, MuinmosSetting, MuinmosToken, OrderAssessment, GuestAccountOTP, GuestLoginSession, SearchHistory, GuestAccountNotificationSetting
from decimal import Decimal
import uuid as _uuid
from datetime import datetime, timezone, timedelta
import urllib.parse
import urllib.request
import json as _json
import smtplib
from email.message import EmailMessage
import os
import boto3
import logging
from enums import UsageStatus

JWT_SECRET = os.getenv("JWT_SECRET", "")
WEBHOOK_TARGET_LAMBDA_ARN = os.getenv("WEBHOOK_TARGET_LAMBDA_ARN", "KYCFastAPIFunctionExternal")
LOGIN_EXPIRY_MINUTES = int(os.getenv("LOGIN_EXPIRY_MINUTES", "1440"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    import stripe
except ImportError:
    stripe = None
try:
    from mangum import Mangum
except Exception:
    Mangum = None

app = FastAPI(title="KYC Backend")

# Adding router to test DB connection
router = APIRouter()

# client Lambda
lambda_client = boto3.client('lambda', region_name=os.getenv('AWS_REGION', 'us-east-1'))

@router.get("/check-db")
def check_db():
    # db_url = os.getenv("DATABASE_URL")
    # if not db_url:
    #     return {"status": "error", "message": "DATABASE_URL not set in Lambda environment"}
    conn_string = os.getenv("CONN_STRING")
    if not conn_string:
        return {"status": "error", "message": "Connection String not set in Lambda environment"}

    try:
        #conn = psycopg2.connect(db_url, connect_timeout=5)
        # For psycopg2, try this format:        
        conn = psycopg2.connect(conn_string, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"status": "success", "postgres_version": version}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Include router to FastAPI App
app.include_router(router)
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://staging.d27qd8fx6l7txq.amplifyapp.com"
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class LoginOTPRequest(BaseModel):
    email: EmailStr
    is_from_login: bool = False

class SubmitOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class CheckoutStartRequest(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    company_name: str
    country_id: str
    service_id: str
    price: float
    currency_id: str
    currency_code: str
    cancel_url: str
    success_url: str

class GuestAccountProfileRequest(BaseModel):
    guest_account_id: str

class AuthValidationRequest(BaseModel):
    guest_account_id: str

class UpdateGuestAccountProfileRequest(BaseModel):
    guest_account_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country_id: Optional[str] = None
    phone: Optional[str] = None
    company_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    zip_postal_code: Optional[str] = None

class UpdateGuestAccountNotificationSettingsRequest(BaseModel):
    guest_account_id: str
    column_name: str
    column_value: bool

class GetOrderPaymentsByGuestAccountRequest(BaseModel):
    guest_account_id: str
    sort_by: str = "transaction_date"
    is_desc: bool = True
    page_size: int = 10
    page_number: int = 1

class GetSearchHistoriesByGuestAccountRequest(BaseModel):
    guest_account_id: str
    sort_by: str = "completed_time"
    is_desc: bool = True
    page_size: int = 10
    page_number: int = 1

class SubmitMuinmosAnswerRequest(BaseModel):
    assessment_id: str
    answer: list

@app.get("/")
def read_root():
    return {"message": "Welcome to KYC Backend API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _load_env_files():
    base_dir = Path(__file__).parent
    candidates = [base_dir / ".env.local", base_dir / ".env"]
    for p in candidates:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    if k:
                        os.environ[k] = v

@app.on_event("startup")
def on_startup():
    _load_env_files()
    env = os.getenv("APP_ENV", "local").lower()
    if env in ("local", "dev"):
        Base.metadata.create_all(bind=engine)
    try:
        if env in ("local", "dev") and engine.dialect.name == "postgresql":
            with engine.connect() as conn:
                # Check and fix orders table
                q = text("SELECT data_type FROM information_schema.columns WHERE table_name='orders' AND column_name='order_id'")
                res = conn.execute(q).scalar()
                if res and res.lower() != "uuid":
                    conn.execute(text("ALTER TABLE orders ALTER COLUMN order_id TYPE uuid USING order_id::uuid"))
                    conn.commit()
                
                # Add base_api_url column to muinmos_settings if it doesn't exist
                q = text("SELECT column_name FROM information_schema.columns WHERE table_name='muinmos_settings' AND column_name='base_api_url'")
                res = conn.execute(q).scalar()
                if not res:
                    conn.execute(text("ALTER TABLE muinmos_settings ADD COLUMN base_api_url VARCHAR NOT NULL DEFAULT ''"))
                    conn.commit()
                
                # Add is_popular column to service_prices if it doesn't exist
                q = text("SELECT column_name FROM information_schema.columns WHERE table_name='service_prices' AND column_name='is_popular'")
                res = conn.execute(q).scalar()
                if not res:
                    conn.execute(text("ALTER TABLE service_prices ADD COLUMN is_popular BOOLEAN NOT NULL DEFAULT false"))
                    conn.commit()
                
                # Alter token column in guest_login_sessions to VARCHAR(500)
                q = text("SELECT character_maximum_length FROM information_schema.columns WHERE table_name='guest_login_sessions' AND column_name='token'")
                res = conn.execute(q).scalar()
                if res and res < 500:
                    conn.execute(text("ALTER TABLE guest_login_sessions ALTER COLUMN token TYPE VARCHAR(500)"))
                    conn.commit()
    except Exception:
        pass

@app.post("/orders")
def create_order(payload: OrderCreate, db: Session = Depends(get_db)):
    obj = Order(
        email=payload.email,
        first_name=payload.first_name,
        last_name=payload.last_name,
        company_name=payload.company_name,
        country=payload.country,
        product_name=payload.product_name,
        price=payload.price,
        period=payload.period,
        description=payload.description,
    )
    db.add(obj)
    db.commit()
    return {"order_id": str(obj.order_id), "status": "created"}

@app.get("/orders")
def list_orders(db: Session = Depends(get_db)):
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

@app.get("/orders/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db)):
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

@app.get("/service-prices")
def list_service_prices(db: Session = Depends(get_db)):
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
def _gen_order_code():
    h = _uuid.uuid4().hex
    return h[0:6] + h[24:30]

@app.post("/checkout/start")
def checkout_start(payload: CheckoutStartRequest, db: Session = Depends(get_db)):
    g = db.query(GuestAccount).filter(GuestAccount.email == payload.email).first()
    if not g:
        g = GuestAccount(
            email=payload.email,
            first_name=payload.first_name,
            last_name=payload.last_name,
            company_name=payload.company_name,
            country_id=payload.country_id,
        )
        db.add(g)
        db.flush()
        
        # Auto-create notification settings
        gans = GuestAccountNotificationSetting(guest_account_id=g.guest_account_id)
        db.add(gans)
    order_code = _gen_order_code()
    op = OrderPayment(
        service_id_ordered=payload.service_id,
        guest_account_id=g.guest_account_id,
        first_name=payload.first_name,
        last_name=payload.last_name,
        company_name=payload.company_name,
        country_id=payload.country_id,
        currency_id=payload.currency_id,
        payment_amount=Decimal(str(payload.price)),
        order_code=order_code,
        checkout_url="",
        checkout_session_status="",
        payment_status="",
        transaction_date=None,
        transaction_expired_date=None,
    )
    db.add(op)
    db.flush()
    ss = db.query(StripeSetting).order_by(StripeSetting.stripe_setting_id.desc()).first()
    secret = ss.secret_key if ss else ""
    amount_for_stripe = int(round(payload.price * 100))
    sp = db.query(ServicePrice).filter(ServicePrice.service_price_id == payload.service_id).first()
    product_name = sp.service_name if sp else "Service"
    form = {
        "success_url": payload.success_url + order_code,
        "cancel_url": payload.cancel_url,
        "mode": "payment",
        "payment_method_types[0]": "card",
        "payment_method_types[1]": "link",
        "payment_method_types[2]": "paynow",
        "line_items[0][price_data][currency]": payload.currency_code,
        "line_items[0][price_data][product_data][name]": product_name,
        "line_items[0][price_data][unit_amount]": str(amount_for_stripe),
        "line_items[0][quantity]": "1",
        "customer_email": payload.email,
        "metadata[internal_order_payment_id]": str(op.order_payment_id),
        "payment_intent_data[metadata][internal_order_payment_id]": str(op.order_payment_id),
    }

    # Invoke an external (non-VPC) Lambda to perform the Stripe Checkout session creation
    if WEBHOOK_TARGET_LAMBDA_ARN:
        lambda_payload = {"action": "create_checkout_session", "payload": form}
        try:
            response = lambda_client.invoke(
                FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                InvocationType="RequestResponse",
                Payload=_json.dumps(lambda_payload).encode("utf-8"),
            )
        except Exception as e:
            logger.exception("checkout: lambda invoke error")
            db.rollback()
            return {
                "status_code": 500,
                "error_message": "Error invoking external payment service",
                "details": str(e),
            }
    else:
        logger.warning("checkout: WEBHOOK_TARGET_LAMBDA_ARN not set")
        db.rollback()
        return {"status_code": 500, "error_message": "External payment service not configured"}

    # 1. Read streaming payload
    payload_bytes = response["Payload"].read()
    payload_str = payload_bytes.decode("utf-8")
    payload_json = json.loads(payload_str)

    logger.info("Lambda B response: %s", payload_json)

    # Parse Lambda response payload (robust for Lambda HTTP shape)
    resp_body = {}
    parsed = payload_json if isinstance(payload_json, dict) else None
    if isinstance(parsed, dict):
        outer_status = parsed.get("statusCode") or parsed.get("StatusCode") or parsed.get("status")
        body = parsed.get("body")
        if body is not None:
            if isinstance(body, str):
                try:
                    resp_body = _json.loads(body)
                except Exception:
                    resp_body = {"body": body}
            elif isinstance(body, dict):
                resp_body = body
            else:
                resp_body = {"body": body}
        else:
            # No body - use the parsed dict directly
            resp_body = parsed

        logger.info("Lambda parsed payload: %s", resp_body)

        # If outer status indicates error, return early
        try:
            sc = int(outer_status) if outer_status is not None else None
            if sc is not None and sc != 200:
                logger.error("checkout: external lambda returned non-200: %s", parsed)
                db.rollback()
                return {"status_code": sc, "error_message": "External payment service returned error", "details": parsed}
        except Exception:
            pass
    else:
        # Not a recognized lambda shape - fallback to raw string
        resp_body = {"body": payload_str}

        logger.info("Lambda fallback response parsed: %s", resp_body)

    # Extract session object from returned structure
    session = None
    if isinstance(resp_body, dict):
        if "session" in resp_body:
            session = resp_body.get("session")
        elif "body" in resp_body:
            body = resp_body.get("body")
            # body may be a JSON string or dict
            if isinstance(body, str):
                try:
                    body_json = _json.loads(body)
                except Exception:
                    body_json = None
            else:
                body_json = body
            if isinstance(body_json, dict) and "session" in body_json:
                session = body_json.get("session")
    # If no session found, treat as error
    if not session:
        logger.error("checkout: no session returned from external lambda: %s", resp_body)
        db.rollback()
        return {"status_code": 500, "error_message": "Error creating checkout session", "details": resp_body}

    # session might be a dict or a JSON string
    if isinstance(session, str):
        try:
            session = _json.loads(session)
        except Exception:
            session = {"id": session}

    # Update order payment with session info
    created_ts = session.get("created")
    expires_ts = session.get("expires_at")
    checkout_url = session.get("url") or session.get("payment_url") or ""
    ref_id = session.get("id")
    op.checkout_url = checkout_url or ""
    op.psp_ref_id = ref_id or ""
    try:
        if created_ts:
            op.transaction_date = datetime.fromtimestamp(int(created_ts), tz=timezone.utc)
    except Exception:
        pass
    try:
        if expires_ts:
            op.transaction_expired_date = datetime.fromtimestamp(int(expires_ts), tz=timezone.utc)
    except Exception:
        pass

    db.commit()

    cur = db.query(Currency).filter(Currency.currency_id == payload.currency_id).first()
    cn = db.query(Country).filter(Country.country_id == payload.country_id).first()
    total_text = (cur.currency_code if cur else "") + (cur.currency_symbol if cur else "") + f"{payload.price:,.2f}"
    subject = f"Enigmatig KYC & AML - Order Confirmation - ({order_code})"
    body = "Thank you for order our service\n" + \
           "Total : " + total_text + "\n" + \
           "Order Code : " + order_code + "\n" + \
           "Full Name : " + payload.first_name + " " + payload.last_name + "\n" + \
           "Company Name : " + payload.company_name + "\n" + \
           "Country : " + (cn.country_name if cn else "")

    # Invoke an external (non-VPC) Lambda to perform the Send Email action
    if WEBHOOK_TARGET_LAMBDA_ARN:
        send_email_payload = {"action": "send_email_smtp", "payload": {"to_email": payload.email, "subject": subject, "body": body, "is_html": False}}
        try:
            response = lambda_client.invoke(
                FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                InvocationType="RequestResponse",
                Payload=_json.dumps(send_email_payload).encode("utf-8"),
            )
        except Exception as e:
            logger.exception("send_email: lambda invoke error")
            return {
                "status_code": 500,
                "error_message": "Error invoking external payment service",
                "details": str(e),
            }
    else:
        logger.warning("send_email: WEBHOOK_TARGET_LAMBDA_ARN not set; skipping email")

    return {
        "status_code": 200,
        "checkout_url": checkout_url,
        "psp_ref_id": ref_id,
        "success_url": payload.success_url,
        "cancel_url": payload.cancel_url,
        #"email_status": ("sent" if email_res.get("ok") else "failed"),
        #"email_reason": email_res.get("reason"),
        #"email_error": email_res.get("error"),
    }

@app.get("/service-prices/{service_price_id}")
def get_service_price(service_price_id: str, db: Session = Depends(get_db)):
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

@app.get("/countries")
def list_countries(db: Session = Depends(get_db)):
    rows = db.query(Country).order_by(Country.country_name.asc()).all()
    return [
        {
            "country_id": str(r.country_id),
            "country_name": r.country_name,
        }
        for r in rows
    ]
@app.get("/order-payments/by-code/{order_code}")
def get_order_payment_by_code(order_code: str, db: Session = Depends(get_db)):
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

def _process_stripe_webhook_event(event: dict, db: Session) -> dict:
    """
    Process Stripe webhook events and update OrderPayment records.
    
    Supported events:
    - payment_intent.succeeded: Payment completed successfully
    - payment_intent.payment_failed: Payment failed
    - checkout.session.completed: Checkout session completed
    - charge.refunded: Payment refunded
    """
    
    event_type = event.get("type", "")
    event_data = event.get("data", {})
    
    # Handle case where event_data might be a string
    if isinstance(event_data, str):
        try:
            event_data = _json.loads(event_data)
        except Exception:
            event_data = {}
    
    result = {
        "event_id": event.get("id"),
        "event_type": event_type,
        "processed": False,
        "message": "",
        "order_payment_id": None,
    }
    try:
        # Extract metadata with order payment ID
        metadata = event_data.get("metadata", {}) if isinstance(event_data, dict) else {}
        order_payment_id = metadata.get("internal_order_payment_id")
                
        if not order_payment_id:
            result["message"] = "No internal_order_payment_id found in metadata"
            return result
        
        # Query the order payment record
        op = db.query(OrderPayment).filter(
            OrderPayment.order_payment_id == order_payment_id
        ).first()
        
        if not op:
            result["message"] = f"OrderPayment not found: {order_payment_id}"
            return result
        
        result["order_payment_id"] = order_payment_id
        
        # Handle different event types
        if event_type == "payment_intent.succeeded":
            pi_id = event_data.get("id", "")
            receipt_url = event_data.get("charges", {}).get("data", [{}])[0].get("receipt_url", "")
            
            op.payment_status = "succeeded"
            op.checkout_session_status = "completed"
            op.psp_stripe_payment_intent = pi_id
            op.psp_stripe_receipt_url = receipt_url
            op.transaction_date = datetime.fromtimestamp(
                event_data.get("created", 0), tz=timezone.utc
            )
            
            result["processed"] = True
            result["message"] = f"Payment succeeded for order {op.order_code}"
            
        elif event_type == "payment_intent.payment_failed":
            pi_id = event_data.get("id", "")
            error_msg = event_data.get("last_payment_error", {}).get("message", "Unknown error")
            
            op.payment_status = "failed"
            op.checkout_session_status = "payment_failed"
            op.psp_stripe_payment_intent = pi_id
            
            result["processed"] = True
            result["message"] = f"Payment failed for order {op.order_code}: {error_msg}"
            
        elif event_type == "checkout.session.completed":            
            session_id = event_data.get("id", "")
            payment_status = event_data.get("payment_status", "")
            payment_intent = event_data.get("payment_intent", "")
            
            op.checkout_session_status = "complete"
            op.psp_ref_id = session_id
            op.psp_stripe_payment_intent = payment_intent
            op.usage_status = UsageStatus.Usable
            
            if payment_status == "paid":
                op.payment_status = "paid"
                result["processed"] = True
                result["message"] = f"Checkout session completed for order {op.order_code}"
            else:
                op.payment_status = payment_status
                result["processed"] = True
                result["message"] = f"Checkout session completed with status: {payment_status}"            

        elif event_type == "charge.refunded":
            charge_id = event_data.get("id", "")
            refund_amount = event_data.get("refunded", 0)
            
            op.payment_status = "refunded"
            op.psp_ref_id = charge_id
            
            result["processed"] = True
            result["message"] = f"Payment refunded for order {op.order_code}"
        
        elif event_type == "charge.updated":
            isPaid = event_data.get("paid", False)
            if isPaid:
                op.psp_stripe_receipt_url = event_data.get("receipt_url", "")
                op.usage_status = UsageStatus.Usable            

            result["processed"] = True
            result["message"] = f"Payment updated for order {op.order_code}"
        
        elif event_type == "checkout.session.expired":
            session_id = event_data.get("id", "")
            payment_status = event_data.get("payment_status", "")
            status = event_data.get("status", "")
            expires_at = event_data.get("expires_at", "")
            
            op.checkout_session_status = status
            op.payment_status = payment_status
            op.psp_ref_id = session_id
            op.usage_status = UsageStatus.Unuseable
            
            if expires_at:
                try:
                    op.transaction_expired_date = datetime.fromtimestamp(int(expires_at), tz=timezone.utc)
                except Exception:
                    pass
            
            result["processed"] = True
            result["message"] = f"Checkout session expired for order {op.order_code}"
        
        else:
            result["message"] = f"Event type '{event_type}' not handled"
        
        # Commit changes if processed
        if result["processed"]:
            db.commit()
            
            # Call create_assessment after commit for checkout.session.completed with paid status
            if event_type == "checkout.session.completed" and event_data.get("payment_status") == "paid":
                guest = db.query(GuestAccount).filter(GuestAccount.guest_account_id == op.guest_account_id).first()
                if guest:
                    assessment_result = create_assessment(guest.email, op.order_code, db)
                    result["assessment"] = assessment_result
        
    except Exception as e:
        db.rollback()
        result["message"] = f"Error processing event: {str(e)}"
    
    return result

@app.get("/muinmos/token")
def get_muinmos_token_endpoint(db: Session = Depends(get_db)):
    return get_muinmos_token(db)

@app.post("/muinmos/create-assessment")
def create_assessment_endpoint(user_email: str, order_code: str, db: Session = Depends(get_db)):
    return create_assessment(user_email, order_code, db)

@app.post("/muinmos/assessment-check")
def muinmos_assessment_check_endpoint(db: Session = Depends(get_db)):
    return muinmos_assessment_check(db)

@app.get("/muinmos/question/{assessment_id}")
def get_muinmos_question_endpoint(assessment_id: str, db: Session = Depends(get_db)):
    return get_muinmos_question(assessment_id, db)

@app.post("/muinmos/submit-answer")
def submit_muinmos_answer_endpoint(request: Request, payload: SubmitMuinmosAnswerRequest, db: Session = Depends(get_db)):
    guest_account_id = request.headers.get("GuestAccountId", "")
    guest_login_token = request.headers.get("GuestLoginToken", "")
    return submit_muinmos_answer(
        guest_account_id,
        guest_login_token,
        payload.assessment_id,
        payload.answer,
        db
    )

def get_muinmos_token(db: Session = Depends(get_db)):
    """Get Muinmos token, fetch new one if expired or doesn't exist"""
    token = db.query(MuinmosToken).first()
    now = datetime.now(timezone.utc)
    
    # Check if token exists and is still valid
    if token and token.expired_at > now:
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expired_at": token.expired_at.isoformat()
        }
    
    # Get Muinmos settings
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    # Invoke external Lambda to get token
    payload = {
        "action": "get_muinmos_token",
        "payload": {
            "grant_type": settings.grant_type,
            "client_id": settings.organization_id,
            "client_secret": settings.client_secret,
            "username": settings.username,
            "password": settings.password,
            "api_url": settings.base_api_url + "/token?api-version=2.0"
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("get_muinmos_token: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        
        logger.info(f"Muinmos token response: {response_payload}")
        
        if "error" in response_payload:
            return {"error": response_payload["error"]}
        
        # Extract token data from response
        token_data = response_payload.get("token_data", response_payload)
        logger.info(f"Extracted token_data: {token_data}")
        
        # Validate required fields
        access_token = token_data.get("access_token")
        token_type = token_data.get("token_type")
        logger.info(f"access_token exists: {bool(access_token)}, token_type: {token_type}")
        
        if not access_token:
            logger.error(f"No access_token in response: {response_payload}")
            return {"error": "No access_token in response from external Lambda"}
        
        # Calculate expiration time (expires_in - 300 seconds buffer)
        expires_in = token_data.get("expires_in", 3600)
        expired_at = now + timedelta(seconds=expires_in - 300)
        
        logger.info(f"About to save token - access_token length: {len(access_token)}, token_type: {token_type}, expires_in: {expires_in}")
        
        if token:
            # Update existing token
            token.access_token = access_token
            token.token_type = token_type
            token.expired_at = expired_at
            token.created_at = now
        else:
            # Insert new token
            token = MuinmosToken(
                access_token=access_token,
                token_type=token_type,
                expired_at=expired_at
            )
            db.add(token)
        
        db.commit()
        
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expired_at": token.expired_at.isoformat()
        }
    
    except Exception as e:
        logger.exception("Error getting Muinmos token")
        return {"error": str(e)}

def create_assessment(user_email: str, order_code: str, db: Session):
    """Create assessment for order"""
    # Step 1: Validate order payment
    op = db.query(OrderPayment).filter(
        OrderPayment.order_code == order_code,
        OrderPayment.usage_status == UsageStatus.Usable,
        OrderPayment.checkout_session_status == "complete",
        OrderPayment.payment_status == "paid"
    ).first()
    
    if not op:
        return {"error": "Not Allowed"}
    
    # Step 2: Check service limits
    sp = db.query(ServicePrice).filter(ServicePrice.service_price_id == op.service_id_ordered).first()
    if not sp:
        return {"error": "Service not found"}
    
    count_oa = db.query(OrderAssessment).filter(OrderAssessment.order_payment_id == op.order_payment_id).count()
    
    if not sp.is_search_by_credit:
        if sp.search_number and sp.search_number > count_oa:
            reference_key = f"{order_code}-{count_oa + 1}"
        else:
            return {"error": "Not Allowed"}
    else:
        if sp.search_number is not None:
            if sp.search_number > count_oa:
                reference_key = f"{order_code}-{count_oa + 1}"
            else:
                return {"error": "Not Allowed"}
        else:
            return {"error": "Not Allowed"}
    
    # Step 3: Get Muinmos settings
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    # Step 4: Get Muinmos token
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
    # Step 5: Invoke external Lambda to create assessment
    payload = {
        "action": "create_assessment",
        "payload": {
            "user_email": user_email,
            "order_code": reference_key,
            "api_url": settings.base_api_url + "/api/assessment?api-version=2.0",
            "token_type": token_response["token_type"],
            "access_token": token_response["access_token"]
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("create_assessment: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        
        if not response_payload.get("success"):
            return {"error": "Failed to create assessment", "details": response_payload}
        
        assessment_id = response_payload.get("assessment_id")
        if not assessment_id:
            return {"error": "No assessment_id in response"}
        
        # Parse assessment_id if it's a JSON string
        if isinstance(assessment_id, str):
            try:
                assessment_id = _json.loads(assessment_id)
            except Exception:
                assessment_id = assessment_id.strip('"')
        
        # Step 6: Insert new order assessment
        oa = OrderAssessment(
            order_payment_id=op.order_payment_id,
            assessment_id=assessment_id,
            reference_key=reference_key,
            pdf_sent=False
        )
        db.add(oa)
        db.commit()
        db.refresh(oa)
        
        # Step 7: Return success
        return {
            "status": "success create an assessment",
            "assessment_id": assessment_id
        }
    
    except Exception as e:
        logger.exception("Error creating assessment")
        db.rollback()
        return {"error": str(e)}

def muinmos_assessment_check(db: Session):
    """Check and update assessment completion status"""
    # Step 1: Get incomplete assessments
    assessments = db.query(OrderAssessment).filter(
        OrderAssessment.is_complete == False
    ).order_by(OrderAssessment.created_date.asc()).all()
    
    if not assessments:
        return {"status": "no_pending_assessments"}
    
    from_date = assessments[0].created_date.isoformat()
    to_date = assessments[-1].created_date.isoformat()
    
    # Step 2: Get Muinmos settings
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    base_api_url = settings.base_api_url
    
    # Step 3: Get Muinmos token
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        logger.error(f"Failed to get Muinmos token: {token_response}")
        return token_response
    
    # Step 4: Invoke external Lambda
    payload = {
        "action": "muinmos_assessment_search",
        "payload": {
            "from_date": from_date,
            "to_date": to_date,
            "base_api_url": base_api_url,
            "token_type": token_response["token_type"],
            "access_token": token_response["access_token"]
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("muinmos_assessment_check: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        
        if not response_payload.get("success"):
            return {"error": "Failed to search assessments", "details": response_payload}
        
        data = response_payload.get("data", {})
        remote_assessments = data.get("assessments", [])
        
        # Step 5: Loop and update
        updated_count = 0
        for oa in assessments:
            matched = next((a for a in remote_assessments if a.get("id") == str(oa.assessment_id)), None)
            
            if not matched:
                continue
            
            if matched.get("state") != "Completed":
                continue
            
            oa.is_complete = True
            db.commit()
            updated_count += 1
            
            # Get order payment and service price
            op = db.query(OrderPayment).filter(
                OrderPayment.order_payment_id == oa.order_payment_id
            ).first()
            
            if not op:
                continue
            
            sp = db.query(ServicePrice).filter(
                ServicePrice.service_price_id == op.service_id_ordered
            ).first()
            
            if not sp or sp.is_search_by_credit:
                continue
            
            count = db.query(OrderAssessment).filter(
                OrderAssessment.order_payment_id == oa.order_payment_id
            ).count()
            
            if count >= sp.search_number:
                op.usage_status = UsageStatus.Completed
                db.commit()
        
        return {
            "status": "success",
            "checked": len(assessments),
            "updated": updated_count
        }
    
    except Exception as e:
        logger.exception("Error checking assessments")
        return {"error": str(e)}

def check_muinmos_assessment_to_send_kycpdf(db: Session):
    """Check completed assessments and send KYC PDF via email"""
    # Step 1: Get completed assessments not yet sent
    query = db.query(
        OrderAssessment.order_assessment_id,
        GuestAccount.email,
        OrderAssessment.assessment_id
    ).join(
        OrderPayment, OrderAssessment.order_payment_id == OrderPayment.order_payment_id
    ).join(
        GuestAccount, OrderPayment.guest_account_id == GuestAccount.guest_account_id
    ).filter(
        OrderAssessment.is_complete == True,
        OrderAssessment.pdf_sent == False
    )
    
    results = query.all()
    
    if not results:
        return {"status": "no_assessments_to_send"}
    
    # Step 2: Get Muinmos settings and token
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        logger.error(f"Failed to get Muinmos token: {token_response}")
        return token_response
    
    assessment_list = [
        {
            "order_assessment_id": str(row.order_assessment_id),
            "email": row.email,
            "assessment_id": str(row.assessment_id)
        }
        for row in results
    ]
    
    # Step 3: Invoke external Lambda
    payload = {
        "action": "send_muinmos_assessment_kycpdf",
        "payload": {
            "base_api_url": settings.base_api_url,
            "token_type": token_response["token_type"],
            "access_token": token_response["access_token"],
            "assessment_list": assessment_list
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("check_muinmos_assessment_to_send_kycpdf: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        
        if not response_payload.get("success"):
            return {"error": "Failed to send KYC PDFs", "details": response_payload}
        
        send_email_result_list = response_payload.get("results", [])
        
        # Step 4: Update pdf_sent status
        updated_count = 0
        for item in send_email_result_list:
            if item.get("is_pdf_sent"):
                oa = db.query(OrderAssessment).filter(
                    OrderAssessment.order_assessment_id == item["order_assessment_id"]
                ).first()
                if oa:
                    oa.pdf_sent = True
                    db.commit()
                    updated_count += 1
        
        return {
            "status": "success",
            "total": len(assessment_list),
            "sent": updated_count
        }
    
    except Exception as e:
        logger.exception("Error sending KYC PDFs")
        return {"error": str(e)}

def update_order_assessment_iscomplete_sendpdfreport(event_type: str, assessment_id: str, reference_key: str, db: Session):
    """Update assessment completion and send PDF report"""
    # Step 1: Update is_complete if event_type is 0
    if event_type == "0":
        query = db.query(
            OrderAssessment,
            GuestAccount.email
        ).join(
            OrderPayment, OrderAssessment.order_payment_id == OrderPayment.order_payment_id
        ).join(
            GuestAccount, OrderPayment.guest_account_id == GuestAccount.guest_account_id
        ).filter(
            OrderAssessment.assessment_id == assessment_id,
            OrderAssessment.reference_key == reference_key
        )
        
        result = query.first()
        
        if not result:
            return {"error": "Assessment not found"}
        
        oa, email = result
        
        # Check if already complete
        if not oa.is_complete:
            oa.is_complete = True
            db.commit()
            
            # Check order payment and update usage_status if needed
            op_check = db.query(
                OrderPayment.order_payment_id,
                OrderPayment.service_id_ordered,
                ServicePrice.is_search_by_credit,
                ServicePrice.search_number
            ).join(
                ServicePrice, OrderPayment.service_id_ordered == ServicePrice.service_price_id
            ).filter(
                OrderPayment.order_payment_id == oa.order_payment_id
            ).first()
            
            if op_check:
                if not op_check.is_search_by_credit:
                    complete_count = db.query(OrderAssessment).filter(
                        OrderAssessment.order_payment_id == op_check.order_payment_id,
                        OrderAssessment.is_complete == True
                    ).count()
                    
                    if complete_count >= op_check.search_number:
                        op = db.query(OrderPayment).filter(
                            OrderPayment.order_payment_id == op_check.order_payment_id
                        ).first()
                        if op:
                            op.usage_status = UsageStatus.Completed
                            db.commit()
        
        # Create search history
        try:
            create_search_history(str(oa.assessment_id), db)
        except Exception:
            logger.exception("Failed to create search history")
        
        # Check if PDF already sent
        if oa.pdf_sent:
            return {"status": "skipped", "reason": "PDF already sent"}
        
        # Step 2: Get Muinmos settings and token
        settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
        if not settings:
            return {"error": "No active Muinmos settings found"}
        
        token_response = get_muinmos_token(db)
        if "error" in token_response:
            logger.error(f"Failed to get Muinmos token: {token_response}")
            return token_response
        
        # Step 3: Invoke external Lambda
        payload = {
            "action": "send_muinmos_assessment_kycpdf_single_user",
            "payload": {
                "base_api_url": settings.base_api_url,
                "token_type": token_response["token_type"],
                "access_token": token_response["access_token"],
                "email": email,
                "assessment_id": str(oa.assessment_id)
            }
        }
        
        if not WEBHOOK_TARGET_LAMBDA_ARN:
            logger.warning("update_order_assessment: WEBHOOK_TARGET_LAMBDA_ARN not set")
            return {"error": "External service not configured"}
        
        try:
            response = lambda_client.invoke(
                FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                InvocationType="RequestResponse",
                Payload=_json.dumps(payload).encode("utf-8")
            )
            response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
            
            if response_payload.get("success") and response_payload.get("is_pdf_sent"):
                oa.pdf_sent = True
                db.commit()
                return {"status": "success", "pdf_sent": True}
            else:
                return {"status": "success", "pdf_sent": False, "details": response_payload}
        
        except Exception as e:
            logger.exception("Error sending PDF report")
            return {"error": str(e)}
    
    return {"status": "skipped", "reason": "event_type is not 0"}

def get_muinmos_question(assessment_id: str, db: Session):
    """Get Muinmos question for assessment"""
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    payload = {
        "action": "get_muinmos_question",
        "payload": {
            "base_api_url": settings.base_api_url,
            "assessment_id": assessment_id
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("get_muinmos_question: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        return response_payload
    except Exception as e:
        logger.exception("Error : " + str(e))
        return {"error": "Failed to get Muinmos question, please contact the administrator."}

def submit_muinmos_answer(guest_account_id: str, guest_account_token: str, assessment_id: str, answer: list, db: Session):
    """Submit Muinmos answer"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] != "valid":
        return JSONResponse(status_code=401, content={"message": "unauthorized"})
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
    payload = {
        "action": "submit_muinmos_answer",
        "payload": {
            "base_api_url": settings.base_api_url,
            "token_type": token_response["token_type"],
            "access_token": token_response["access_token"],
            "assessment_id": assessment_id,
            "answer": answer
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("submit_muinmos_answer: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        response_payload["token_expiry_on"] = auth_result["expiry_on"]
        return response_payload
    except Exception as e:
        logger.exception("Error : " + str(e))
        return {"error": "Failed to submit answer, please contact the administrator."}

def gen_login_otp():
    """Generate 4-digit OTP"""
    return str(random.randint(1000, 9999))

def gen_login_token(email: str):
    """Generate JWT login token"""
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "email": email,
            "iat": int(now.timestamp()),
            "jti": str(uuid4())
        },
        JWT_SECRET,
        algorithm="HS256"
    )
    return token

@app.post("/auth/email-get-otp")
def login_with_email_generate_otp_endpoint(payload: LoginOTPRequest, db: Session = Depends(get_db)):
    return login_with_email_generate_otp(payload.email, payload.is_from_login, db)

@app.post("/auth/submit-otp")
def login_submit_otp_endpoint(payload: SubmitOTPRequest, db: Session = Depends(get_db)):
    return login_submit_otp(payload.email, payload.otp, db)

def login_with_email_generate_otp(email: str, is_from_login: bool, db: Session):
    """Generate and send OTP for email login"""
    is_send_email = False
    otp = gen_login_otp()
    
    # Get or create guest account
    ga = db.query(GuestAccount).filter(GuestAccount.email == email).first()
    
    if ga:
        # Check existing OTP
        gaotp = db.query(GuestAccountOTP).filter(
            GuestAccountOTP.guest_account_id == ga.guest_account_id
        ).first()
        
        if gaotp:
            now = datetime.now(timezone.utc)
            if gaotp.expiry_date <= now or is_from_login:
                gaotp.requested_date = datetime.now(timezone.utc)
                gaotp.otp = otp
                gaotp.expiry_date = datetime.now(timezone.utc) + timedelta(minutes=5)
                db.commit()
                is_send_email = True
        else:
            now = datetime.now(timezone.utc)
            gaotp = GuestAccountOTP(
                guest_account_id=ga.guest_account_id,
                requested_date=now,
                otp=otp,
                expiry_date=now + timedelta(minutes=5)
            )
            db.add(gaotp)
            db.commit()
            is_send_email = True
    else:
        ga = GuestAccount(email=email)
        db.add(ga)
        db.flush()
        
        # Auto-create notification settings
        gans = GuestAccountNotificationSetting(guest_account_id=ga.guest_account_id)
        db.add(gans)
        
        now = datetime.now(timezone.utc)
        gaotp = GuestAccountOTP(
            guest_account_id=ga.guest_account_id,
            requested_date=now,
            otp=otp,
            expiry_date=now + timedelta(minutes=5)
        )
        db.add(gaotp)
        db.commit()
        is_send_email = True
    
    # Send email if needed
    if is_send_email:
        if WEBHOOK_TARGET_LAMBDA_ARN:
            payload = {
                "action": "send_email_smtp",
                "payload": {
                    "to_email": ga.email,
                    "subject": f"Enigmatig KYC & AML - {otp} is your personal code.",
                    "body": f"Hi,<br/><br/>Your personal unique code is: {otp}.<br/>Please type your code into the login box to connect to your account.",
                    "is_html": True
                }
            }
            
            try:
                lambda_client.invoke(
                    FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                    InvocationType="RequestResponse",
                    Payload=_json.dumps(payload).encode("utf-8")
                )
            except Exception as e:
                logger.exception("Error sending OTP email")
        else:
            logger.warning("login_with_email_generate_otp: WEBHOOK_TARGET_LAMBDA_ARN not set; skipping OTP email")
    
    return {"guest_account_id": str(ga.guest_account_id)}

def login_submit_otp(email: str, otp: str, db: Session):
    """Submit OTP and generate login token"""
    # Get guest account
    ga = db.query(GuestAccount).filter(GuestAccount.email == email).first()
    
    if not ga:
        return JSONResponse(
            status_code=401,
            content={"message": "The account is not found."}
        )
    
    # Get OTP record
    gaotp = db.query(GuestAccountOTP).filter(
        GuestAccountOTP.guest_account_id == ga.guest_account_id
    ).first()
    
    if not gaotp:
        return JSONResponse(
            status_code=410,
            content={"message": "The code is incorrect. Please check your email."}
        )
    
    # Check OTP match
    if gaotp.otp != otp:
        return JSONResponse(
            status_code=410,
            content={"message": "The code is incorrect. Please check your email."}
        )
    
    # Check expiry
    now = datetime.now(timezone.utc)
    if gaotp.expiry_date < now:
        return JSONResponse(
            status_code=410,
            content={"message": "OTP has been expired."}
        )
    
    # Generate token
    token = gen_login_token(email)
    expiry_on = now + timedelta(minutes=LOGIN_EXPIRY_MINUTES)
    
    # Update or create login session
    gls = db.query(GuestLoginSession).filter(
        GuestLoginSession.guest_account_id == ga.guest_account_id
    ).first()
    
    if gls:
        gls.issued_on = now
        gls.expiry_on = expiry_on
        gls.token = token
    else:
        gls = GuestLoginSession(
            guest_account_id=ga.guest_account_id,
            issued_on=now,
            expiry_on=expiry_on,
            token=token
        )
        db.add(gls)
    
    db.commit()
    
    # Calculate expiry_on in Unix timestamp (seconds)
    expiry_timestamp = int(expiry_on.timestamp())
    
    return {
        "message": "Success.",
        "token": token,
        "expiry_on": expiry_timestamp
    }

def auth_validation_by_token_and_guest_account_id(guest_account_id: str, token: str, db: Session):
    """Validate authentication token and extend expiry if valid"""
    result = {
        "auth_status": "",
        "token": token,
        "expiry_on": ""
    }
    
    gls = db.query(GuestLoginSession).filter(
        GuestLoginSession.guest_account_id == guest_account_id,
        GuestLoginSession.token == token
    ).first()
    
    if not gls:
        result["auth_status"] = "not found"
    else:
        now = datetime.now(timezone.utc)
        if gls.expiry_on > now:
            expiry_on = now + timedelta(minutes=LOGIN_EXPIRY_MINUTES)
            gls.expiry_on = expiry_on
            db.commit()
            result["auth_status"] = "valid"
            result["expiry_on"] = int(expiry_on.timestamp())
        else:
            result["auth_status"] = "expired"
            result["expiry_on"] = int(gls.expiry_on.timestamp())
    
    return result

@app.post("/auth/validate-by-token-and-guest-account-id")
def auth_validation_endpoint(request: Request, payload: AuthValidationRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return auth_validation_by_token_and_guest_account_id(payload.guest_account_id, guest_account_token, db)

@app.get("/service-info/{order_code}")
def get_service_info_by_order_code_endpoint(order_code: str, request: Request, db: Session = Depends(get_db)):
    guest_account_id = request.headers.get("GuestAccountId", "")
    guest_login_token = request.headers.get("GuestLoginToken", "")
    return get_service_info_by_order_code(order_code, guest_account_id, guest_login_token, db)

def get_service_info_by_order_code(order_code: str, guest_account_id: str, token: str, db: Session):
    """Get service info by order code"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, token, db)
    
    if auth_result["auth_status"] == "valid":
        # Get order payment with joins
        result = db.query(
            OrderPayment.order_payment_id,
            OrderPayment.order_code,
            OrderPayment.payment_status,
            OrderPayment.checkout_session_status,
            OrderPayment.usage_status,
            ServicePrice.service_price_id,
            ServicePrice.service_name,
            ServicePrice.is_search_by_credit,
            ServicePrice.search_number,
            GuestAccount.guest_account_id,
            GuestAccount.email
        ).join(
            ServicePrice, OrderPayment.service_id_ordered == ServicePrice.service_price_id
        ).join(
            GuestAccount, OrderPayment.guest_account_id == GuestAccount.guest_account_id
        ).filter(
            OrderPayment.order_code == order_code,
            OrderPayment.guest_account_id == guest_account_id
        ).first()
        
        if result:
            # Get order assessments
            assessments = db.query(OrderAssessment).filter(
                OrderAssessment.order_payment_id == result.order_payment_id
            ).all()
            
            assessment_list = [
                {
                    "order_assessment_id": str(oa.order_assessment_id),
                    "order_payment_id": str(oa.order_payment_id),
                    "assessment_id": str(oa.assessment_id),
                    "is_complete": oa.is_complete,
                    "pdf_sent": oa.pdf_sent,
                    "reference_key": oa.reference_key
                }
                for oa in assessments
            ]
            
            order_payment_info = {
                "guest_account_id": str(result.guest_account_id),
                "email": result.email,
                "order_payment_id": str(result.order_payment_id),
                "order_code": result.order_code,
                "payment_status": result.payment_status,
                "checkout_session_status": result.checkout_session_status,
                "usage_status": result.usage_status,
                "service_price_id": str(result.service_price_id),
                "service_name": result.service_name,
                "is_search_by_credit": result.is_search_by_credit,
                "search_number": result.search_number,
                "order_assessments": assessment_list
            }
            
            return JSONResponse(status_code=200, content={
                "order_code_info": order_payment_info,
                "assessment_count": len(assessments),
                "token_expiry_on": auth_result["expiry_on"],
            })
        else:
            return JSONResponse(status_code=204, content={"message": "Order Code is not founded."})
    else:
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

@app.post("/guest-account/profile")
def get_guest_account_profile_endpoint(request: Request, payload: GuestAccountProfileRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return get_guest_account_profile(payload.guest_account_id, guest_account_token, db)

def get_guest_account_profile(guest_account_id: str, guest_account_token: str, db: Session):
    """Get guest account profile with notification settings"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        ga = db.query(GuestAccount).filter(GuestAccount.guest_account_id == guest_account_id).first()
        
        if ga:
            gans = db.query(GuestAccountNotificationSetting).filter(
                GuestAccountNotificationSetting.guest_account_id == guest_account_id
            ).first()
            
            guest_account_result = {
                "message": "success",
                "token_expiry_on": auth_result["expiry_on"],
                "guest_account": {
                    "guest_account_id": str(ga.guest_account_id),
                    "email": ga.email,
                    "first_name": ga.first_name,
                    "last_name": ga.last_name,
                    "company_name": ga.company_name,
                    "country_id": str(ga.country_id) if ga.country_id else None,
                    "address": ga.address,
                    "city": ga.city,
                    "zip_postal_code": ga.zip_postal_code,
                    "phone": ga.phone,
                    "created_at": ga.created_at.isoformat() if ga.created_at else None
                }
            }
            
            if gans:
                guest_account_result["guest_account_notification_setting"] = {
                    "guest_account_notification_setting_id": str(gans.guest_account_notification_setting_id),
                    "guest_account_id": str(gans.guest_account_id),
                    "email_promotion_subscription": gans.email_promotion_subscription,
                    "email_system_messages": gans.email_system_messages,
                    "phone_promotion_subscription": gans.phone_promotion_subscription,
                    "sms_system_messages": gans.sms_system_messages
                }
            
            return JSONResponse(status_code=200, content=guest_account_result)
        else:
            return JSONResponse(status_code=500, content={"message": "guest_account_id is not founded"})
    else:
        return JSONResponse(status_code=500, content={"message": "unauthorized"})

@app.post("/guest-account/update-profile")
def update_guest_account_profile_endpoint(request: Request, payload: UpdateGuestAccountProfileRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return update_guest_account_profile(
        payload.guest_account_id,
        guest_account_token,
        payload.first_name,
        payload.last_name,
        payload.country_id,
        payload.phone,
        payload.company_name,
        payload.address,
        payload.city,
        payload.zip_postal_code,
        db
    )

def update_guest_account_profile(guest_account_id: str, guest_account_token: str, first_name: str, last_name: str, country_id: str, phone: str, company_name: str, address: str, city: str, zip_postal_code: str, db: Session):
    """Update guest account profile"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        try:
            ga = db.query(GuestAccount).filter(GuestAccount.guest_account_id == guest_account_id).first()
            
            if not ga:
                return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})
            
            if first_name is not None:
                ga.first_name = first_name
            if last_name is not None:
                ga.last_name = last_name
            if country_id is not None:
                ga.country_id = country_id
            if phone is not None:
                ga.phone = phone
            if company_name is not None:
                ga.company_name = company_name
            if address is not None:
                ga.address = address
            if city is not None:
                ga.city = city
            if zip_postal_code is not None:
                ga.zip_postal_code = zip_postal_code
            
            db.commit()
            
            return JSONResponse(status_code=200, content={"message": "success", "token_expiry_on": auth_result["expiry_on"]})
        
        except Exception as e:
            db.rollback()
            logger.exception("Error updating guest account profile")
            if "timeout" in str(e).lower():
                return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
            else:
                return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})
    else:
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

@app.post("/guest-account/update-notification-settings")
def update_guest_account_notification_settings_endpoint(request: Request, payload: UpdateGuestAccountNotificationSettingsRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return update_guest_account_notification_settings(
        payload.guest_account_id,
        guest_account_token,
        payload.column_name,
        payload.column_value,
        db
    )

def update_guest_account_notification_settings(guest_account_id: str, guest_account_token: str, column_name: str, column_value: bool, db: Session):
    """Update guest account notification settings"""
    # Get or create notification settings
    gans = db.query(GuestAccountNotificationSetting).filter(
        GuestAccountNotificationSetting.guest_account_id == guest_account_id
    ).first()
    
    if not gans:
        gans = GuestAccountNotificationSetting(
            guest_account_id=guest_account_id,
            email_promotion_subscription=False,
            email_system_messages=False,
            phone_promotion_subscription=False,
            sms_system_messages=False
        )
        setattr(gans, column_name, column_value)
        db.add(gans)
        db.commit()
    
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        try:
            setattr(gans, column_name, column_value)
            db.commit()
            return JSONResponse(status_code=200, content={"message": "success", "token_expiry_on": auth_result["expiry_on"]})
        except Exception as e:
            db.rollback()
            logger.exception("Error updating notification settings")
            if "timeout" in str(e).lower():
                return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
            else:
                return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})
    else:
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

@app.post("/guest-account/order-payments")
def get_order_payments_by_guest_account_endpoint(request: Request, payload: GetOrderPaymentsByGuestAccountRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    logger.info(f"get_order_payments_by_guest_account_endpoint called with guest_account_id: {payload.guest_account_id}, sort_by: {payload.sort_by}, is_desc: {payload.is_desc}, page_size: {payload.page_size}, page_number: {payload.page_number}")
    return get_order_payments_by_guest_account(
        payload.guest_account_id,
        guest_account_token,
        payload.sort_by,
        payload.is_desc,
        payload.page_size,
        payload.page_number,
        db
    )

def get_order_payments_by_guest_account(guest_account_id: str, guest_account_token: str, sort_by: str, is_desc: bool, page_size: int, page_number: int, db: Session):
    """Get order payments by guest account with pagination"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        # Step 1: Get total count (no join needed)
        total_count = db.query(OrderPayment).filter(
            OrderPayment.guest_account_id == guest_account_id
        ).count()
        
        # Step 2: Get paginated data with join
        sort_column = getattr(OrderPayment, sort_by, OrderPayment.transaction_date)
        query = db.query(
            OrderPayment.transaction_date,
            OrderPayment.transaction_expired_date,
            OrderPayment.service_id_ordered,
            ServicePrice.service_name,
            OrderPayment.order_code,
            OrderPayment.usage_status,
            OrderPayment.payment_status,
            OrderPayment.checkout_session_status,
            OrderPayment.checkout_url,
            OrderPayment.psp_stripe_receipt_url
        ).join(
            ServicePrice, OrderPayment.service_id_ordered == ServicePrice.service_price_id
        ).filter(
            OrderPayment.guest_account_id == guest_account_id
        )
        
        if is_desc:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        results = query.offset((page_number - 1) * page_size).limit(page_size).all()
        
        data_list = [
            {
                "transaction_date": row.transaction_date.isoformat() if row.transaction_date else None,
                "transaction_expired_date": row.transaction_expired_date.isoformat() if row.transaction_expired_date else None,
                "service_id_ordered": str(row.service_id_ordered) if row.service_id_ordered else None,
                "service_name": row.service_name,
                "order_code": row.order_code,
                "usage_status": row.usage_status,
                "payment_status": row.payment_status,
                "checkout_session_status": row.checkout_session_status,
                "checkout_url": row.checkout_url,
                "psp_stripe_receipt_url": row.psp_stripe_receipt_url
            }
            for row in results
        ]
        
        is_has_more = (page_number * page_size) < total_count
        
        return JSONResponse(status_code=200, content={
            "token_expiry_on": auth_result["expiry_on"],
            "data_list": data_list,
            "total_count": total_count,
            "is_has_more": is_has_more,
            "current_page_number": page_number
        })
    else:
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

@app.post("/guest-account/search-histories")
def get_search_histories_by_guest_account_id_endpoint(request: Request, payload: GetSearchHistoriesByGuestAccountRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return get_search_histories_by_guest_account_id(
        payload.guest_account_id,
        guest_account_token,
        payload.sort_by,
        payload.is_desc,
        payload.page_size,
        payload.page_number,
        db
    )

def get_search_histories_by_guest_account_id(guest_account_id: str, guest_account_token: str, sort_by: str, is_desc: bool, page_size: int, page_number: int, db: Session):
    """Get search histories by guest account with pagination"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        # Step 1: Get total count
        total_count = db.query(SearchHistory).join(
            OrderAssessment, SearchHistory.order_assessment_id == OrderAssessment.order_assessment_id
        ).join(
            OrderPayment, OrderAssessment.order_payment_id == OrderPayment.order_payment_id
        ).join(
            GuestAccount, OrderPayment.guest_account_id == GuestAccount.guest_account_id
        ).filter(
            GuestAccount.guest_account_id == guest_account_id
        ).count()
        
        # Step 2: Get paginated data with join
        sort_column = getattr(SearchHistory, sort_by, SearchHistory.completed_time)
        query = db.query(
            SearchHistory.completed_time,
            OrderAssessment.reference_key,
            SearchHistory.first_name,
            SearchHistory.middle_name,
            SearchHistory.last_name,
            SearchHistory.dob,
            SearchHistory.rag_result,
            OrderAssessment.pdf_sent
        ).join(
            OrderAssessment, SearchHistory.order_assessment_id == OrderAssessment.order_assessment_id
        ).join(
            OrderPayment, OrderAssessment.order_payment_id == OrderPayment.order_payment_id
        ).join(
            GuestAccount, OrderPayment.guest_account_id == GuestAccount.guest_account_id
        ).filter(
            GuestAccount.guest_account_id == guest_account_id
        )
        
        if is_desc:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        results = query.offset((page_number - 1) * page_size).limit(page_size).all()
        
        data_list = [
            {
                "completed_time": row.completed_time.isoformat() if row.completed_time else None,
                "reference_key": row.reference_key,
                "first_name": row.first_name,
                "middle_name": row.middle_name,
                "last_name": row.last_name,
                "dob": row.dob.isoformat() if row.dob else None,
                "rag_result": row.rag_result,
                "pdf_sent": row.pdf_sent
            }
            for row in results
        ]
        
        is_has_more = (page_number * page_size) < total_count
        
        return JSONResponse(status_code=200, content={
            "token_expiry_on": auth_result["expiry_on"],
            "data_list": data_list,
            "total_count": total_count,
            "is_has_more": is_has_more,
            "current_page_number": page_number
        })
    else:
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

def create_search_history(assessment_id: str, db: Session):
    """Create search history record"""
    oa = db.query(OrderAssessment).filter(OrderAssessment.assessment_id == assessment_id).first()
    if not oa:
        return {"error": "Assessment not found"}
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
    payload = {
        "action": "get_muinmos_assessment_result",
        "payload": {
            "base_api_url": settings.base_api_url,
            "token_type": token_response["token_type"],
            "access_token": token_response["access_token"],
            "assessment_id": assessment_id
        }
    }
    
    if not WEBHOOK_TARGET_LAMBDA_ARN:
        logger.warning("create_search_history: WEBHOOK_TARGET_LAMBDA_ARN not set")
        return {"error": "External service not configured"}
    
    try:
        response = lambda_client.invoke(
            FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=_json.dumps(payload).encode("utf-8")
        )
        response_payload = _json.loads(response["Payload"].read().decode("utf-8"))
        
        if not response_payload.get("success"):
            return {"error": "Failed to get assessment result", "details": response_payload}
        
        completed_time = response_payload.get("completed_time")
        rag_result = response_payload.get("ragResult", "")
        answers = response_payload.get("answers", {})
        
        completed_dt = None
        if completed_time:
            try:
                completed_dt = datetime.fromisoformat(completed_time.replace("Z", "+00:00"))
            except Exception:
                pass
        
        dob_dt = None
        if answers.get("dob"):
            try:
                dob_dt = datetime.fromisoformat(answers["dob"].replace("Z", "+00:00"))
            except Exception:
                pass
        
        sh = SearchHistory(
            order_assessment_id=oa.order_assessment_id,
            completed_time=completed_dt,
            first_name=answers.get("first_name"),
            middle_name=answers.get("middle_name"),
            last_name=answers.get("last_name"),
            dob=dob_dt,
            rag_result=rag_result
        )
        db.add(sh)
        db.commit()
        db.refresh(sh)
        
        return {"search_history_id": str(sh.search_history_id)}
    
    except Exception as e:
        logger.exception("Error creating search history")
        db.rollback()
        return {"error": str(e)}
# Test auto deployment #2
if Mangum:
    handler = Mangum(app, lifespan="off")
