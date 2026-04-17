from pydantic import BaseModel
from typing import Optional

class GuestAccountProfileRequest(BaseModel):
    guest_account_id: str

class UpdateGuestAccountProfileRequest(BaseModel):
    guest_account_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country_id: Optional[str] = None
    phone: Optional[str] = None
    company_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    zip_postal_code: Optional[str] = None

class UpdateGuestAccountNotificationSettingsRequest(BaseModel):
    guest_account_id: str
    column_name: str
    column_value: bool

class GetOrderPaymentsByGuestAccountRequest(BaseModel):
    guest_account_id: str
    sort_by: str = "transaction_date"
    is_desc: bool = True
    page_size: int = 10
    page_number: int = 1

class GetSearchHistoriesByGuestAccountRequest(BaseModel):
    guest_account_id: str
    sort_by: str = "completed_time"
    is_desc: bool = True
    page_size: int = 10
    page_number: int = 1

class CreateReferralCodeRequest(BaseModel):
    guest_account_id: str

class GetReferralCodeRequest(BaseModel):
    guest_account_id: str

class ApplyReferralCodeRequest(BaseModel):
    guest_account_id: str
    referral_code: str

class GetReferredUsersRequest(BaseModel):
    guest_account_id: str
    sort_by: str = "created_at"
    is_desc: bool = True
    page_size: int = 10
    page_number: int = 1
