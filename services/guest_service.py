import logging
import re
import random
import string
import uuid
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi.responses import JSONResponse

from models import (
    GuestAccount, GuestAccountNotificationSetting, OrderPayment, 
    ServicePrice, SearchHistory, OrderAssessment, GuestAccountReferral
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

# Referral Code Functions

def clean_prefix(email: str) -> str:
    """
    Extract and clean prefix from email for referral code generation.
    
    Rules:
    - Take substring before '@'
    - Remove '.', '-', '_'
    - Take maximum 5 characters
    - If less than 5 characters, keep as-is (do NOT pad)
    
    Args:
        email: User's email address
        
    Returns:
        Cleaned prefix string (max 5 chars)
        
    Examples:
        ss@mail.com → ss
        afs@mail.com → afs
        qwe.sdfg@mail.co.id → qwesd
        adinda.fitria123@gmail.com → adind
    """
    if not email or '@' not in email:
        raise ValueError("Invalid email format")
    
    # Extract part before '@'
    prefix = email.split('@')[0]
    
    # Remove special characters
    prefix = re.sub(r'[.\-_]', '', prefix)
    
    # Take maximum 5 characters
    prefix = prefix[:5]
    
    # Convert to lowercase for consistency
    prefix = prefix.lower()
    
    if not prefix:
        raise ValueError("Email prefix cannot be empty after cleaning")
    
    return prefix

def generate_code(prefix: str) -> str:
    """
    Generate referral code with given prefix.
    
    Format: <prefix><2 digit number><4 alphanumeric characters>
    
    Args:
        prefix: Cleaned email prefix (max 5 chars)
        
    Returns:
        Complete referral code
        
    Examples:
        ss → ss23a45g
        afs → afs02bf43
        qwesd → qwesd12100b
    """
    # Generate 2-digit number (00-99)
    two_digit = f"{random.randint(0, 99):02d}"
    
    # Generate 4 alphanumeric characters (lowercase letters + digits)
    chars = string.ascii_lowercase + string.digits
    four_chars = ''.join(random.choice(chars) for _ in range(4))
    
    # Combine all parts
    referral_code = f"{prefix}{two_digit}{four_chars}"
    
    return referral_code

def create_referral_code(guest_account_id: str, guest_account_token: str, db: Session) -> dict:
    """
    Generate unique referral code and insert into database.
    
    This function:
    1. Validates authentication token
    2. Gets email from GuestAccount table
    3. Generates referral code based on email
    4. Inserts directly into database (no pre-check)
    5. Handles unique constraint violations by retrying
    6. Returns the final successful referral code
    
    Args:
        guest_account_id: UUID of the guest account
        guest_account_token: Authentication token for the guest account
        db: Database session
        
    Returns:
        Dictionary with referral code and token expiry info
        
    Raises:
        ValueError: Invalid input parameters or guest account not found
        RuntimeError: Failed to generate unique code after max retries
    """
    # Validate authentication first
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] != "valid":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"message": "unauthorized"})
    
    if not guest_account_id:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"message": "guest_account_id is required", "token_expiry_on": auth_result["expiry_on"]})
    
    # Validate UUID format
    try:
        uuid.UUID(guest_account_id)
    except ValueError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"message": "Invalid guest_account_id UUID format", "token_expiry_on": auth_result["expiry_on"]})
    
    try:
        # Get email from GuestAccount
        guest_account = db.query(GuestAccount).filter(
            GuestAccount.guest_account_id == guest_account_id
        ).first()
        
        if not guest_account:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"message": "Guest account not found", "token_expiry_on": auth_result["expiry_on"]})
        
        if not guest_account.email:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"message": "Guest account email is required", "token_expiry_on": auth_result["expiry_on"]})
        
        # Clean email prefix
        prefix = clean_prefix(guest_account.email)
        
        max_retries = 10
        attempt = 0
        
        while attempt < max_retries:
            attempt += 1
            
            # Generate new referral code
            referral_code = generate_code(prefix)
            
            try:
                # Check if GuestAccountReferral already exists for this guest_account_id
                existing_referral = db.query(GuestAccountReferral).filter(
                    GuestAccountReferral.guest_account_id == guest_account_id
                ).first()
                
                if existing_referral:
                    # Update existing record
                    existing_referral.referral_code = referral_code
                    logger.info(f"Referral code updated successfully: {referral_code} for guest {guest_account_id}")
                else:
                    # Insert new record
                    referral_record = GuestAccountReferral(
                        guest_account_id=guest_account_id,
                        referral_code=referral_code
                    )
                    db.add(referral_record)
                    logger.info(f"Referral code created successfully: {referral_code} for guest {guest_account_id}")
                
                db.commit()
                
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=200, content={
                    "message": "success",
                    "referral_code": referral_code,
                    "token_expiry_on": auth_result["expiry_on"]
                })
                        
            except IntegrityError as e:
                # Referral code already exists, retry with new code
                db.rollback()
                logger.warning(f"Referral code collision on attempt {attempt}: {referral_code}")
                if attempt >= max_retries:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=500, content={
                        "message": "Failed to generate unique referral code after maximum attempts",
                        "token_expiry_on": auth_result["expiry_on"]
                    })
                continue
                
            except Exception as e:
                db.rollback()
                logger.error(f"Database error on attempt {attempt}: {str(e)}")
                from fastapi.responses import JSONResponse
                if "timeout" in str(e).lower():
                    return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
                else:
                    return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})
        
        # This should never be reached due to the logic above, but included for completeness
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={
            "message": "Failed to generate unique referral code after maximum attempts",
            "token_expiry_on": auth_result["expiry_on"]
        })
        
    except Exception as e:
        db.rollback()
        logger.exception("Error creating referral code")
        from fastapi.responses import JSONResponse
        if "timeout" in str(e).lower():
            return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
        else:
            return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})

def get_referral_code(guest_account_id: str, guest_account_token: str, db: Session) -> dict:
    """
    Retrieve existing referral code for a guest account.
    
    Args:
        guest_account_id: UUID of the guest account
        guest_account_token: Authentication token for the guest account
        db: Database session
        
    Returns:
        Dictionary with referral code and token expiry info
        
    Raises:
        ValueError: Invalid guest_account_id
    """
    # Validate authentication first
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] != "valid":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"message": "unauthorized"})
    
    if not guest_account_id:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"message": "guest_account_id is required", "token_expiry_on": auth_result["expiry_on"]})
    
    try:
        uuid.UUID(guest_account_id)
    except ValueError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"message": "Invalid guest_account_id UUID format", "token_expiry_on": auth_result["expiry_on"]})
    
    try:
        referral = db.query(GuestAccountReferral).filter(
            GuestAccountReferral.guest_account_id == guest_account_id
        ).order_by(GuestAccountReferral.created_at.desc()).first()
        
        if referral and referral.referral_code:
            response_data = {
                "message": "success",
                "referral_code": referral.referral_code,
                "referred_by_id": None,
                "referred_by": None,
                "token_expiry_on": auth_result["expiry_on"]
            }
            
            # Check if referred_by_id is not null
            if referral.referred_by_id:
                # Get the referral code of the referrer
                referrer = db.query(GuestAccountReferral).filter(
                    GuestAccountReferral.guest_account_referral_id == referral.referred_by_id
                ).first()
                
                if referrer:
                    response_data["referred_by_id"] = str(referral.referred_by_id)
                    response_data["referred_by"] = referrer.referral_code
            
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=200, content=response_data)
        else:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={
                "message": "No referral code found",
                "token_expiry_on": auth_result["expiry_on"]
            })
            
    except Exception as e:
        logger.exception("Error retrieving referral code")
        from fastapi.responses import JSONResponse
        if "timeout" in str(e).lower():
            return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
        else:
            return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})

def apply_referral_code_by_auth_guest_account(guest_account_id: str, guest_account_token: str, referral_code: str, db: Session) -> dict:
    """Apply a referral code to an authenticated guest account."""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)

    if auth_result["auth_status"] != "valid":
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

    if not guest_account_id:
        return JSONResponse(status_code=400, content={"message": "guest_account_id is required", "token_expiry_on": auth_result["expiry_on"]})

    if not referral_code or not referral_code.strip():
        return JSONResponse(status_code=400, content={"message": "referral_code is required", "token_expiry_on": auth_result["expiry_on"]})

    try:
        uuid.UUID(guest_account_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"message": "Invalid guest_account_id UUID format", "token_expiry_on": auth_result["expiry_on"]})

    try:
        referred_by = db.query(GuestAccountReferral).filter(
            GuestAccountReferral.referral_code == referral_code.strip()
        ).first()

        if not referred_by:
            return JSONResponse(status_code=404, content={"message": "Referral code is not found.", "token_expiry_on": auth_result["expiry_on"]})

        existing_referral = db.query(GuestAccountReferral).filter(
            GuestAccountReferral.guest_account_id == guest_account_id
        ).order_by(GuestAccountReferral.created_at.desc()).first()

        if existing_referral:
            existing_referral.referred_by_id = referred_by.guest_account_referral_id
        else:
            new_referral = GuestAccountReferral(
                guest_account_id=guest_account_id,
                referral_code=None,
                referred_by_id=referred_by.guest_account_referral_id,
            )
            db.add(new_referral)

        db.commit()

        return JSONResponse(
            status_code=200, 
            content={
                "message": "success", 
                "referred_by": referred_by.referral_code,
                "referred_by_id": str(referred_by.guest_account_referral_id),
                "token_expiry_on": auth_result["expiry_on"]
                }
            )
    except Exception as e:
        db.rollback()
        logger.exception("Error applying referral code")
        if "timeout" in str(e).lower():
            return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
        else:
            return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})

def get_referred_users(guest_account_id: str, guest_account_token: str, sort_by: str, is_desc: bool, page_size: int, page_number: int, db: Session):
    """Get list of users referred by this guest account with pagination"""
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)

    if auth_result["auth_status"] != "valid":
        return JSONResponse(status_code=401, content={"message": "unauthorized"})

    try:
        gar = db.query(GuestAccountReferral).filter(
            GuestAccountReferral.guest_account_id == guest_account_id
        ).first()

        if not gar:
            return JSONResponse(status_code=200, content={
                "token_expiry_on": auth_result["expiry_on"],
                "data_list": [],
                "total_count": 0,
                "is_has_more": False,
                "current_page_number": page_number
            })

        total_count = db.query(GuestAccountReferral).filter(
            GuestAccountReferral.referred_by_id == gar.guest_account_referral_id
        ).count()

        sort_column = getattr(GuestAccountReferral, sort_by, GuestAccountReferral.created_at)
        query = db.query(
            GuestAccountReferral.created_at,
            GuestAccount.email
        ).join(
            GuestAccount, GuestAccountReferral.guest_account_id == GuestAccount.guest_account_id
        ).filter(
            GuestAccountReferral.referred_by_id == gar.guest_account_referral_id
        )

        if is_desc:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        results = query.offset((page_number - 1) * page_size).limit(page_size).all()

        data_list = [
            {
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "email": row.email
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
    except Exception as e:
        logger.exception("Error retrieving referred users")
        if "timeout" in str(e).lower():
            return JSONResponse(status_code=408, content={"message": "timeout", "token_expiry_on": auth_result["expiry_on"]})
        else:
            return JSONResponse(status_code=500, content={"message": "failed", "token_expiry_on": auth_result["expiry_on"]})

def _get_referral_code_internal(guest_account_id: str, db: Session) -> str:
    """
    Internal helper function to retrieve referral code without authentication.
    Used by get_or_create_referral_code after authentication is already validated.
    
    Args:
        guest_account_id: UUID of the guest account
        db: Database session
        
    Returns:
        Referral code if exists, None otherwise
    """
    referral = db.query(GuestAccountReferral).filter(
        GuestAccountReferral.guest_account_id == guest_account_id
    ).order_by(GuestAccountReferral.created_at.desc()).first()
    
    return referral.referral_code if referral else None

def get_or_create_referral_code(guest_account_id: str, guest_account_token: str, db: Session) -> dict:
    """
    Get existing referral code or create new one if doesn't exist.
    
    Args:
        guest_account_id: UUID of the guest account
        guest_account_token: Authentication token for the guest account
        db: Database session
        
    Returns:
        Dictionary with referral code and token expiry info
    """
    # Validate authentication first
    auth_result = auth_validation_by_token_and_guest_account_id(guest_account_id, guest_account_token, db)
    
    if auth_result["auth_status"] != "valid":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"message": "unauthorized"})
    
    # Try to get existing referral code first
    existing_code = _get_referral_code_internal(guest_account_id, db)
    
    if existing_code:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=200, content={
            "message": "success",
            "referral_code": existing_code,
            "token_expiry_on": auth_result["expiry_on"]
        })
    
    # Create new referral code if none exists
    return create_referral_code(guest_account_id, guest_account_token, db)
