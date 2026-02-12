from mangum import Mangum
from main import app, _process_stripe_webhook_event, muinmos_assessment_check, check_muinmos_assessment_to_send_kycpdf, update_order_assessment_iscomplete_sendpdfreport, SessionLocal

def lambda_handler(event, context):
    """Direct Lambda handler for non-HTTP invocations"""
    if event.get("action") == "process_webhook":
        db = SessionLocal()
        try:
            webhook_event = event.get("webhook_event", {})
            result = _process_stripe_webhook_event(webhook_event, db)
            return result
        finally:
            db.close()
    elif event.get("action") == "update_order_assessment_iscomplete_sendpdfreport":
        db = SessionLocal()
        try:
            event_type = event.get("event_type", "")
            assessment_id = event.get("assessment_id", "")
            reference_key = event.get("reference_key", "")
            result = update_order_assessment_iscomplete_sendpdfreport(event_type, assessment_id, reference_key, db)
            return result
        finally:
            db.close()
    elif "detail" in event and "action" in event.get("detail", {}):
        # EventBridge scheduled event with custom input
        action = event["detail"]["action"]
        db = SessionLocal()
        try:
            if action == "muinmos_assessment_check":
                result = muinmos_assessment_check(db)
            elif action == "check_muinmos_assessment_to_send_kycpdf":
                result = check_muinmos_assessment_to_send_kycpdf(db)
            else:
                result = {"error": f"Unknown EventBridge action: {action}"}
            return result
        finally:
            db.close()
    else:
        # Use Mangum for HTTP requests
        mangum_handler = Mangum(app, api_gateway_base_path="/dev")
        return mangum_handler(event, context)

handler = lambda_handler