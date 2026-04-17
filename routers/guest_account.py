from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from db import SessionLocal
from schemas.guest_schemas import (
    GuestAccountProfileRequest,
    UpdateGuestAccountProfileRequest,
    UpdateGuestAccountNotificationSettingsRequest,
    GetOrderPaymentsByGuestAccountRequest,
    GetSearchHistoriesByGuestAccountRequest,
    CreateReferralCodeRequest,
    GetReferralCodeRequest,
    ApplyReferralCodeRequest,
    GetReferredUsersRequest
)
from services.guest_service import (
    get_guest_account_profile,
    update_guest_account_profile,
    update_guest_account_notification_settings,
    get_order_payments_by_guest_account,
    get_search_histories_by_guest_account_id,
    create_referral_code,
    get_referral_code,
    apply_referral_code_by_auth_guest_account,
    get_referred_users
)

router = APIRouter(prefix="/guest-account", tags=["guest_account"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/profile")
def get_guest_account_profile_endpoint(request: Request, payload: GuestAccountProfileRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return get_guest_account_profile(payload.guest_account_id, guest_account_token, db)

@router.post("/update-profile")
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

@router.post("/update-notification-settings")
def update_guest_account_notification_settings_endpoint(request: Request, payload: UpdateGuestAccountNotificationSettingsRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return update_guest_account_notification_settings(
        payload.guest_account_id,
        guest_account_token,
        payload.column_name,
        payload.column_value,
        db
    )

@router.post("/order-payments")
def get_order_payments_by_guest_account_endpoint(request: Request, payload: GetOrderPaymentsByGuestAccountRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return get_order_payments_by_guest_account(
        payload.guest_account_id,
        guest_account_token,
        payload.sort_by,
        payload.is_desc,
        payload.page_size,
        payload.page_number,
        db
    )

@router.post("/search-histories")
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

@router.post("/request-referral-code")
def request_referral_code_endpoint(request: Request, payload: CreateReferralCodeRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return create_referral_code(payload.guest_account_id, guest_account_token, db)

@router.post("/get-referral-code")
def get_referral_code_endpoint(request: Request, payload: GetReferralCodeRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return get_referral_code(payload.guest_account_id, guest_account_token, db)

@router.post("/apply-referral-code")
def apply_referral_code_endpoint(request: Request, payload: ApplyReferralCodeRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return apply_referral_code_by_auth_guest_account(
        payload.guest_account_id,
        guest_account_token,
        payload.referral_code,
        db
    )

@router.post("/referred-users")
def get_referred_users_endpoint(request: Request, payload: GetReferredUsersRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return get_referred_users(
        payload.guest_account_id,
        guest_account_token,
        payload.sort_by,
        payload.is_desc,
        payload.page_size,
        payload.page_number,
        db
    )
