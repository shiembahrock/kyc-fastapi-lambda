from sqlalchemy import Column, String, Text, DateTime, Integer, Numeric, CheckConstraint, Boolean, ForeignKey
from sqlalchemy.sql import func
from db import Base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from enums import UsageStatus

class Order(Base):
    __tablename__ = "orders"
    order_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    company_name = Column(String(255), nullable=False)
    country = Column(String(120), nullable=False)
    product_name = Column(String(255), nullable=False)
    price = Column(String(50))
    period = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Country(Base):
    __tablename__ = "countries"
    country_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    numeric_code = Column(Integer)
    country_name = Column(String(255), nullable=False)
    alpha_2_code = Column(String(2), nullable=False)
    alpha_3_code = Column(String(3), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Currency(Base):
    __tablename__ = "currencies"
    currency_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    currency_name = Column(String(100), nullable=False)
    currency_code = Column(String(5), nullable=False)
    currency_symbol = Column(String(10), nullable=False)
    exchange_rate = Column(Numeric(30, 12), nullable=False)
    currency_precision = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    __table_args__ = (
        CheckConstraint("exchange_rate >= 0 AND exchange_rate <= 1000000000000", name="chk_exchange_rate_bounds"),
        CheckConstraint("currency_precision >= 0 AND currency_precision <= 10", name="chk_currency_precision_bounds"),
    )

class ServicePrice(Base):
    __tablename__ = "service_prices"
    service_price_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(200), nullable=False)
    price = Column(Numeric(20, 2), nullable=False)
    sort_order = Column(Integer)
    currency = Column(UUID(as_uuid=True), ForeignKey("currencies.currency_id"), nullable=False)
    is_show = Column(Boolean, nullable=False, default=True)
    is_search_by_credit = Column(Boolean, nullable=False, default=False)
    search_number = Column(Integer, nullable=True)
    is_popular = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    __table_args__ = (
        CheckConstraint("price >= -100000000000 AND price <= 100000000000", name="chk_service_price_bounds"),
    )

class GuestAccount(Base):
    __tablename__ = "guest_accounts"
    guest_account_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    company_name = Column(String(255))
    country_id = Column(UUID(as_uuid=True), ForeignKey("countries.country_id"), nullable=True)
    address = Column(String(255))
    city = Column(String(120))
    zip_postal_code = Column(String(40))
    phone = Column(String(40))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class OrderPayment(Base):
    __tablename__ = "order_payments"
    order_payment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_id_ordered = Column(UUID(as_uuid=True), ForeignKey("service_prices.service_price_id"), nullable=True)
    guest_account_id = Column(UUID(as_uuid=True), ForeignKey("guest_accounts.guest_account_id"), nullable=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    company_name = Column(String(255))
    address = Column(String(255))
    city = Column(String(120))
    phone = Column(String(40))
    zip_postal_code = Column(String(40))
    country_id = Column(UUID(as_uuid=True), ForeignKey("countries.country_id"), nullable=True)
    currency_id = Column(UUID(as_uuid=True), ForeignKey("currencies.currency_id"), nullable=True)
    payment_amount = Column(Numeric(20, 2))
    order_code = Column(String(20))
    checkout_url = Column(String(500))
    checkout_session_status = Column(String(50))
    payment_status = Column(String(50))
    transaction_date = Column(DateTime(timezone=True))
    transaction_expired_date = Column(DateTime(timezone=True))
    usage_status = Column(Integer, default=int(UsageStatus.Unuseable), nullable=False)
    psp_ref_id = Column(String(255))
    psp_stripe_payment_intent = Column(String(255))
    psp_stripe_receipt_url = Column(String(500))

class StripeSetting(Base):
    __tablename__ = "stripe_settings"
    stripe_setting_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    publishable_key = Column(String(255))
    secret_key = Column(String(255), nullable=False)

class MuinmosSetting(Base):
    __tablename__ = "muinmos_settings"
    muinmoss_setting_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(String, nullable=False)
    client_secret = Column(String, nullable=False)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)
    grant_type = Column(String, nullable=False)
    base_api_url = Column(String, nullable=False)
    is_used = Column(Boolean, nullable=False, default=True)

class MuinmosToken(Base):
    __tablename__ = "muinmos_tokens"
    muinmos_token_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expired_at = Column(DateTime(timezone=True), nullable=False)
    token_type = Column(String(10))
    access_token = Column(String(1200), nullable=False)

class OrderAssessment(Base):
    __tablename__ = "order_assessments"
    order_assessment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_payment_id = Column(UUID(as_uuid=True), ForeignKey("order_payments.order_payment_id"), nullable=False)
    created_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    assessment_id = Column(UUID(as_uuid=True), nullable=False)
    reference_key = Column(String(100))
    is_complete = Column(Boolean, nullable=False, default=False)
    pdf_sent = Column(Boolean, nullable=False, default=False)

class GuestAccountOTP(Base):
    __tablename__ = "guest_account_otps"
    guest_account_otp_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_account_id = Column(UUID(as_uuid=True), ForeignKey("guest_accounts.guest_account_id"), nullable=False)
    requested_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expiry_date = Column(DateTime(timezone=True), nullable=False)
    otp = Column(String(6))
    is_logged_in = Column(Boolean, nullable=False, default=False)

class GuestLoginSession(Base):
    __tablename__ = "guest_login_sessions"
    guest_login_session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_account_id = Column(UUID(as_uuid=True), ForeignKey("guest_accounts.guest_account_id"), nullable=False)
    issued_on = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expiry_on = Column(DateTime(timezone=True), nullable=False)
    token = Column(String(500), nullable=False)
