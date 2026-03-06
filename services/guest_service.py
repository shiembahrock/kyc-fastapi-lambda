import logging
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse

from models import (
    GuestAccount, GuestAccountNotificationSetting, OrderPayment, 
    ServicePrice, SearchHistory, OrderAssessment
)
from services.auth_service import auth_validation_by_token_and_guest_account_id

logger = logging.getLogger()

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

def update_guest_account_notification_settings(guest_account_id: str, guest_account_token: str, column_name: str, column_value: bool, db: Session):
    """Update guest account notification settings"""
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

def get_order_payments_by_guest_account(guest_account_id: str, guest_account_token: str, sort_by: str, is_desc: bool, page_size: int, page_number: int, db: Session):
    """Get order payments by guest account with pagination"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        total_count = db.query(OrderPayment).filter(
            OrderPayment.guest_account_id == guest_account_id
        ).count()
        
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

def get_search_histories_by_guest_account_id(guest_account_id: str, guest_account_token: str, sort_by: str, is_desc: bool, page_size: int, page_number: int, db: Session):
    """Get search histories by guest account with pagination"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] == "valid":
        total_count = db.query(SearchHistory).join(
            OrderAssessment, SearchHistory.order_assessment_id == OrderAssessment.order_assessment_id
        ).join(
            OrderPayment, OrderAssessment.order_payment_id == OrderPayment.order_payment_id
        ).join(
            GuestAccount, OrderPayment.guest_account_id == GuestAccount.guest_account_id
        ).filter(
            GuestAccount.guest_account_id == guest_account_id
        ).count()
        
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

def get_service_info_by_order_code(order_code: str, guest_account_id: str, token: str, db: Session):
    """Get service info by order code"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, token, db)
    
    if auth_result["auth_status"] == "valid":
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
