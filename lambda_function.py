from mangum import Mangum
from main import app, _process_stripe_webhook_event, muinmos_assessment_check, SessionLocal

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
    elif "detail" in event and "action" in event.get("detail", {}):
        # EventBridge scheduled event with custom input
        action = event["detail"]["action"]
        db = SessionLocal()
        try:
            if action == "muinmos_assessment_check":
                result = muinmos_assessment_check(db)
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