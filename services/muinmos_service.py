import os
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
import json as _json

from models import (
    MuinmosSetting, MuinmosToken, OrderPayment, OrderAssessment, 
    ServicePrice, GuestAccount, SearchHistory
)
from enums import UsageStatus
from utils.lambda_client import lambda_client

WEBHOOK_TARGET_LAMBDA_ARN = os.getenv("WEBHOOK_TARGET_LAMBDA_ARN", "KYCFastAPIFunctionExternal")
logger = logging.getLogger()

def get_muinmos_token(db: Session):
    """Get Muinmos token, fetch new one if expired or doesn't exist"""
    token = db.query(MuinmosToken).first()
    now = datetime.now(timezone.utc)
    
    if token and token.expired_at > now:
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expired_at": token.expired_at.isoformat()
        }
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
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
        
        if "error" in response_payload:
            return {"error": response_payload["error"]}
        
        token_data = response_payload.get("token_data", response_payload)
        access_token = token_data.get("access_token")
        token_type = token_data.get("token_type")
        
        if not access_token:
            return {"error": "No access_token in response from external Lambda"}
        
        expires_in = token_data.get("expires_in", 3600)
        expired_at = now + timedelta(seconds=expires_in - 300)
        
        if token:
            token.access_token = access_token
            token.token_type = token_type
            token.expired_at = expired_at
            token.created_at = now
        else:
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
    op = db.query(OrderPayment).filter(
        OrderPayment.order_code == order_code,
        OrderPayment.usage_status == UsageStatus.Usable,
        OrderPayment.checkout_session_status == "complete",
        OrderPayment.payment_status == "paid"
    ).first()
    
    if not op:
        return {"error": "Not Allowed"}
    
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
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
    payload = {
        "action": "create_assessment",
        "payload": {
            "user_email": user_email,
            "kyc_profile_id": sp.kyc_profile_id,
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
        
        if isinstance(assessment_id, str):
            try:
                assessment_id = _json.loads(assessment_id)
            except Exception:
                assessment_id = assessment_id.strip('"')
        
        oa = OrderAssessment(
            order_payment_id=op.order_payment_id,
            assessment_id=assessment_id,
            reference_key=reference_key,
            pdf_sent=False
        )
        db.add(oa)
        db.commit()
        db.refresh(oa)
        
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
    assessments = db.query(OrderAssessment).filter(
        OrderAssessment.is_complete == False
    ).order_by(OrderAssessment.created_date.asc()).all()
    
    if not assessments:
        return {"status": "no_pending_assessments"}
    
    from_date = assessments[0].created_date.isoformat()
    to_date = assessments[-1].created_date.isoformat()
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
    payload = {
        "action": "muinmos_assessment_search",
        "payload": {
            "from_date": from_date,
            "to_date": to_date,
            "base_api_url": settings.base_api_url,
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
        
        updated_count = 0
        for oa in assessments:
            matched = next((a for a in remote_assessments if a.get("id") == str(oa.assessment_id)), None)
            
            if not matched or matched.get("state") != "Completed":
                continue
            
            oa.is_complete = True
            db.commit()
            updated_count += 1
            
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
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
    assessment_list = [
        {
            "order_assessment_id": str(row.order_assessment_id),
            "email": row.email,
            "assessment_id": str(row.assessment_id)
        }
        for row in results
    ]
    
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
    from services.search_service import create_search_history
    
    if event_type != "0":
        return {"status": "skipped", "reason": "event_type is not 0"}
    
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
    
    if not oa.is_complete:
        oa.is_complete = True
        db.commit()
        
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
    
    try:
        create_search_history(str(oa.assessment_id), db)
    except Exception:
        logger.exception("Failed to create search history")
    
    if oa.pdf_sent:
        return {"status": "skipped", "reason": "PDF already sent"}
    
    settings = db.query(MuinmosSetting).filter(MuinmosSetting.is_used == True).first()
    if not settings:
        return {"error": "No active Muinmos settings found"}
    
    token_response = get_muinmos_token(db)
    if "error" in token_response:
        return token_response
    
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
    from services.auth_service import auth_validation_by_token_and_guest_account_id
    
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
        
        # Check if submission completed and trigger PDF report
        status_code = response_payload.get("statusCode")
        if status_code == 200:
            body = response_payload.get("body", {})
            result = body.get("result", "")
            if isinstance(result, str) and result.lower() == "completed":
                oa = db.query(OrderAssessment).filter(
                    OrderAssessment.assessment_id == assessment_id
                ).first()
                if oa:
                    update_order_assessment_iscomplete_sendpdfreport("0", assessment_id, oa.reference_key, db)
        
        return response_payload
    except Exception as e:
        logger.exception("Error : " + str(e))
        return {"error": "Failed to submit answer, please contact the administrator."}

def create_muinmos_assessment_by_guest_account(guest_account_id: str, guest_account_token: str, order_code: str, db: Session):
    """Create Muinmos assessment by guest account"""
    from services.auth_service import auth_validation_by_token_and_guest_account_id
    
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        ga = db.query(GuestAccount).filter(GuestAccount.guest_account_id == guest_account_id).first()
        if not ga:
            return JSONResponse(status_code=401, content={"message": "unauthorized"})
        
        result = create_assessment(ga.email, order_code, db)
        if isinstance(result, dict) and "error" not in result:
            result["token_expiry_on"] = auth_result["expiry_on"]
        return result
    else:
        return JSONResponse(status_code=401, content={"message": "unauthorized"})
