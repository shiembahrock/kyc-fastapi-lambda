import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
import json as _json

from models import OrderAssessment, MuinmosSetting, SearchHistory
from utils.lambda_client import lambda_client

WEBHOOK_TARGET_LAMBDA_ARN = os.getenv("WEBHOOK_TARGET_LAMBDA_ARN", "KYCFastAPIFunctionExternal")
logger = logging.getLogger()

def create_search_history(assessment_id: str, db: Session):
    """Create search history record"""
    from services.muinmos_service import get_muinmos_token
    
    oa = db.query(OrderAssessment).filter(OrderAssessment.assessment_id == assessment_id).first()
    if not oa:
        return {"error": "Assessment not found"}
    
    # Check if search history already exists
    existing_sh = db.query(SearchHistory).filter(
        SearchHistory.order_assessment_id == oa.order_assessment_id
    ).first()
    
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
        
        if existing_sh:
            # Update existing record
            existing_sh.completed_time = completed_dt
            existing_sh.first_name = answers.get("first_name")
            existing_sh.middle_name = answers.get("middle_name")
            existing_sh.last_name = answers.get("last_name")
            existing_sh.dob = dob_dt
            existing_sh.rag_result = rag_result
            db.commit()
            db.refresh(existing_sh)
            return {"search_history_id": str(existing_sh.search_history_id)}
        else:
            # Insert new record
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
