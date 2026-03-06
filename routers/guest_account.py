from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from db import SessionLocal
from schemas.guest_schemas import (
    GuestAccountProfileRequest,
    UpdateGuestAccountProfileRequest,
    UpdateGuestAccountNotificationSettingsRequest,
    GetOrderPaymentsByGuestAccountRequest,
    GetSearchHistoriesByGuestAccountRequest
)
from services.guest_service import (
    get_guest_account_profile,
    update_guest_account_profile,
    update_guest_account_notification_settings,
    get_order_payments_by_guest_account,
    get_search_histories_by_guest_account_id
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
