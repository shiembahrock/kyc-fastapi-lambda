from fastapi import FastAPI, APIRouter, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import uuid4
from pathlib import Path
import json
from sqlalchemy.orm import Session
from sqlalchemy import text
from db import Base, engine, SessionLocal
from models import Order, ServicePrice, Currency, Country, OrderPayment, GuestAccount, StripeSetting
from decimal import Decimal
import uuid as _uuid
from datetime import datetime, timezone
import urllib.parse
import urllib.request
import json as _json
import smtplib
from email.message import EmailMessage
import os
import boto3
import logging

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
                q = text("SELECT data_type FROM information_schema.columns WHERE table_name='orders' AND column_name='order_id'")
                res = conn.execute(q).scalar()
                if res and res.lower() != "uuid":
                    conn.execute(text("ALTER TABLE orders ALTER COLUMN order_id TYPE uuid USING order_id::uuid"))
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
        }
        for sp, cur in rows
    ]
def _gen_order_code():
    h = _uuid.uuid4().hex
    return h[0:6] + h[24:30]
def _send_email(to_email: str, subject: str, body: str):
    try:
        host = os.getenv("SMTP_HOST", "smtp.office365.com")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASS", "")
        msg = EmailMessage()
        msg["From"] = user or "no-reply@example.com"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)
        if host and port and user and password:
            with smtplib.SMTP(host, port) as s:
                s.starttls()
                s.login(user, password)
                s.send_message(msg)
            return {"ok": True}
        else:
            return {"ok": False, "reason": "missing_config", "details": {"host": host, "port": port, "user": bool(user), "password": bool(password)}}
    except smtplib.SMTPAuthenticationError as e:
        return {"ok": False, "reason": "auth", "error": str(e)}
    except Exception as e:
        return {"ok": False, "reason": "smtp", "error": str(e)}
def _send_email_verbose(to_email: str, subject: str, body: str):
    host = os.getenv("SMTP_HOST", "smtp.office365.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")
    msg = EmailMessage()
    msg["From"] = user or "no-reply@example.com"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    if not (host and port and user and password):
        return {"ok": False, "reason": "missing_config", "details": {"host": host, "port": port, "user": bool(user), "password": bool(password)}}
    try:
        with smtplib.SMTP(host, port) as s:
            c1, m1 = s.ehlo()
            c2, m2 = s.starttls()
            c3, m3 = s.login(user, password)
            refused = s.sendmail(msg["From"], [to_email], msg.as_string())
            return {
                "ok": not bool(refused),
                "ehlo_code": c1,
                "ehlo_msg": (m1.decode("utf-8", errors="ignore") if isinstance(m1, bytes) else str(m1)),
                "starttls_code": c2,
                "starttls_msg": (m2.decode("utf-8", errors="ignore") if isinstance(m2, bytes) else str(m2)),
                "login_code": c3,
                "login_msg": (m3.decode("utf-8", errors="ignore") if isinstance(m3, bytes) else str(m3)),
                "sendmail_refused": refused,
            }
    except smtplib.SMTPAuthenticationError as e:
        return {"ok": False, "reason": "auth", "error": str(e)}
    except Exception as e:
        return {"ok": False, "reason": "smtp", "error": str(e)}
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
    lambda_payload = {"action": "create_checkout_session", "payload": form}
    try:
        response = lambda_client.invoke(
            FunctionName="KYCFastAPIFunctionExternal",
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

    # 1. Read streaming payload once (do not re-read, it will exhaust the stream).
    try:
        payload_bytes = response["Payload"].read()
        payload_str = payload_bytes.decode("utf-8") if payload_bytes else ""
        payload_json = _json.loads(payload_str) if payload_str else {}
    except Exception as e:
        logger.exception("checkout: error reading lambda payload")
        db.rollback()
        return {"status_code": 500, "error_message": "Error reading external service response", "details": str(e)}

    logger.info("Lambda B response: %s", payload_json)

    # Invoke status check
    try:
        sc_invoke = response.get("StatusCode") if isinstance(response, dict) else None
        if sc_invoke and int(sc_invoke) != 200:
            logger.error("checkout: lambda invoke failed with StatusCode %s", sc_invoke)
            db.rollback()
            return {"status_code": sc_invoke, "error_message": "External payment service invoke failed", "details": response}
    except Exception:
        # non-fatal - continue to attempt parsing payload
        pass

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
    #email_res = _send_email(payload.email, subject, body)

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
@app.get("/email/test")
def email_test(to: EmailStr, subject: Optional[str] = None, body: Optional[str] = None):
    s = subject or "Test Email"
    b = body or "Test email from KYC Backend"
    res = _send_email_verbose(str(to), s, b)
    return res
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
    event_data = event.get("data", {}).get("object", {})
    
    result = {
        "event_id": event.get("id"),
        "event_type": event_type,
        "processed": False,
        "message": "",
        "order_payment_id": None,
    }
    
    try:
        # Extract metadata with order payment ID
        metadata = event_data.get("metadata", {})
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
            
            op.checkout_session_status = "completed"
            op.psp_ref_id = session_id
            
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
            
        else:
            result["message"] = f"Event type '{event_type}' not handled"
        
        # Commit changes if processed
        if result["processed"]:
            db.commit()
        
    except Exception as e:
        db.rollback()
        result["message"] = f"Error processing event: {str(e)}"
    
    return result

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handle Stripe webhook events.
    
    To verify the webhook signature, set STRIPE_WEBHOOK_SECRET in environment variables.
    Signature verification is optional but highly recommended for production.
    """
    if not stripe:
        return {"error": "stripe library not installed", "status_code": 400}
    
    # Get Stripe configuration
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    stripe_key = os.getenv("STRIPE_SECRET_KEY", "")
    
    if stripe_key:
        stripe.api_key = stripe_key
    
    # Read the request body
    body = await request.body()
    
    # Verify webhook signature if secret is configured
    if webhook_secret:
        sig_header = request.headers.get("stripe-signature", "")
        try:
            event = stripe.Webhook.construct_event(
                body, sig_header, webhook_secret
            )
        except stripe.error.SignatureVerificationError as e:
            return {
                "error": "Invalid signature",
                "details": str(e),
                "status_code": 403,
            }
        except Exception as e:
            return {
                "error": "Error verifying webhook signature",
                "details": str(e),
                "status_code": 400,
            }
    else:
        # If no webhook secret configured, parse the raw body
        try:
            event = _json.loads(body.decode("utf-8"))
        except Exception as e:
            return {
                "error": "Invalid JSON",
                "details": str(e),
                "status_code": 400,
            }
    
    # Process the webhook event
    result = _process_stripe_webhook_event(event, db)
    
    return {
        "received": True,
        "status_code": 200,
        "event_id": result.get("event_id"),
        "event_type": result.get("event_type"),
        "processed": result.get("processed"),
        "message": result.get("message"),
        "order_payment_id": result.get("order_payment_id"),
    }

if Mangum:
    handler = Mangum(app, lifespan="off")
