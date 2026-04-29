import os
import logging
import json
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.orm import Session
import json as _json

from models import OrderPayment, GuestAccount, StripeSetting, ServicePrice, Currency, Country, GuestAccountNotificationSetting, GuestAccountReferral
from enums import UsageStatus
from utils.lambda_client import lambda_client
from utils.helpers import gen_order_code

WEBHOOK_TARGET_LAMBDA_ARN = os.getenv("WEBHOOK_TARGET_LAMBDA_ARN", "KYCFastAPIFunctionExternal")
logger = logging.getLogger()

def process_stripe_webhook_event(event: dict, db: Session) -> dict:
    """Process Stripe webhook events and update OrderPayment records"""
    from services.muinmos_service import create_assessment
    
    event_type = event.get("type", "")
    event_data = event.get("data", {})
    
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
        metadata = event_data.get("metadata", {}) if isinstance(event_data, dict) else {}
        order_payment_id = metadata.get("internal_order_payment_id")
                
        if not order_payment_id:
            result["message"] = "No internal_order_payment_id found in metadata"
            return result
        
        op = db.query(OrderPayment).filter(
            OrderPayment.order_payment_id == order_payment_id
        ).first()
        
        if not op:
            result["message"] = f"OrderPayment not found: {order_payment_id}"
            return result
        
        result["order_payment_id"] = order_payment_id
        
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
        
        if result["processed"]:
            db.commit()
            
            if event_type == "checkout.session.completed" and event_data.get("payment_status") == "paid":
                guest = db.query(GuestAccount).filter(GuestAccount.guest_account_id == op.guest_account_id).first()
                if guest:
                    assessment_result = create_assessment(guest.email, op.order_code, db)
                    result["assessment"] = assessment_result
        
    except Exception as e:
        db.rollback()
        result["message"] = f"Error processing event: {str(e)}"
    
    return result

def checkout_start(email: str, first_name: str, last_name: str, company_name: str, country_id: str, phone: str, service_id: str, price: float, currency_id: str, currency_code: str, cancel_url: str, success_url: str, db: Session):
    """Start checkout process"""
    g = db.query(GuestAccount).filter(GuestAccount.email == email).first()
    if not g:
        g = GuestAccount(
            email=email,
            first_name=first_name,
            last_name=last_name,
            company_name=company_name,
            country_id=country_id,
            phone=phone,
        )
        db.add(g)
        db.flush()
        
        gans = GuestAccountNotificationSetting(guest_account_id=g.guest_account_id)
        db.add(gans)
    else:
        # Update empty/null values with parameters
        if not g.first_name and first_name:
            g.first_name = first_name
        if not g.last_name and last_name:
            g.last_name = last_name
        if not g.company_name and company_name:
            g.company_name = company_name
        if not g.country_id and country_id:
            g.country_id = country_id
        if not g.phone and phone:
            g.phone = phone
    
    order_code = gen_order_code()
    op = OrderPayment(
        service_id_ordered=service_id,
        guest_account_id=g.guest_account_id,
        first_name=first_name,
        last_name=last_name,
        company_name=company_name,
        country_id=country_id,
        phone=phone,
        currency_id=currency_id,
        payment_amount=Decimal(str(price)),
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
    amount_for_stripe = int(round(price * 100))
    sp = db.query(ServicePrice).filter(ServicePrice.service_price_id == service_id).first()
    product_name = sp.stripe_product_id if (sp and sp.stripe_product_id) else (sp.service_name if sp else "Service")
    
    form = {
        "success_url": success_url + order_code,
        "cancel_url": cancel_url,
        "mode": "payment",
        "payment_method_types[0]": "card",
        "payment_method_types[1]": "link",
        "payment_method_types[2]": "paynow",
        "line_items[0][price_data][currency]": currency_code,
        "line_items[0][price_data][product_data][name]": product_name,
        "line_items[0][price_data][unit_amount]": str(amount_for_stripe),
        "line_items[0][quantity]": "1",
        "customer_email": email,
        "metadata[internal_order_payment_id]": str(op.order_payment_id),
        "payment_intent_data[metadata][internal_order_payment_id]": str(op.order_payment_id),
    }

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

    payload_bytes = response["Payload"].read()
    payload_str = payload_bytes.decode("utf-8")
    payload_json = json.loads(payload_str)

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
            resp_body = parsed

        try:
            sc = int(outer_status) if outer_status is not None else None
            if sc is not None and sc != 200:
                logger.error("checkout: external lambda returned non-200: %s", parsed)
                db.rollback()
                return {"status_code": sc, "error_message": "External payment service returned error", "details": parsed}
        except Exception:
            pass
    else:
        resp_body = {"body": payload_str}

    session = None
    if isinstance(resp_body, dict):
        if "session" in resp_body:
            session = resp_body.get("session")
        elif "body" in resp_body:
            body = resp_body.get("body")
            if isinstance(body, str):
                try:
                    body_json = _json.loads(body)
                except Exception:
                    body_json = None
            else:
                body_json = body
            if isinstance(body_json, dict) and "session" in body_json:
                session = body_json.get("session")
    
    if not session:
        logger.error("checkout: no session returned from external lambda: %s", resp_body)
        db.rollback()
        return {"status_code": 500, "error_message": "Error creating checkout session", "details": resp_body}

    if isinstance(session, str):
        try:
            session = _json.loads(session)
        except Exception:
            session = {"id": session}

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

    cur = db.query(Currency).filter(Currency.currency_id == currency_id).first()
    cn = db.query(Country).filter(Country.country_id == country_id).first()
    total_text = (cur.currency_code if cur else "") + (cur.currency_symbol if cur else "") + f"{price:,.2f}"
    subject = f"Enigmatig KYC & AML - Order Confirmation - ({order_code})"
    body = "Thank you for order our service\\n" + \
           "Total : " + total_text + "\\n" + \
           "Order Code : " + order_code + "\\n" + \
           "Full Name : " + first_name + " " + last_name + "\\n" + \
           "Company Name : " + company_name + "\\n" + \
           "Country : " + (cn.country_name if cn else "")

    if WEBHOOK_TARGET_LAMBDA_ARN:
        send_email_payload = {"action": "send_email_smtp", "payload": {"to_email": email, "subject": subject, "body": body, "is_html": False}}
        try:
            lambda_client.invoke(
                FunctionName=WEBHOOK_TARGET_LAMBDA_ARN,
                InvocationType="RequestResponse",
                Payload=_json.dumps(send_email_payload).encode("utf-8"),
            )
        except Exception as e:
            logger.exception("send_email: lambda invoke error")
    else:
        logger.warning("send_email: WEBHOOK_TARGET_LAMBDA_ARN not set; skipping email")

    return {
        "status_code": 200,
        "checkout_url": checkout_url,
        "psp_ref_id": ref_id,
        "success_url": success_url,
        "cancel_url": cancel_url,
    }
