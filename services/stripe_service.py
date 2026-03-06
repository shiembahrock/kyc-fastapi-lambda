import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import json as _json

from models import OrderPayment, GuestAccount
from enums import UsageStatus

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
