# Modular Refactoring Guide

## Current Status
- Created directory structure: routers/, schemas/, services/, utils/
- Created schema files: auth_schemas.py, muinmos_schemas.py, guest_schemas.py, order_schemas.py
- Created utility files: lambda_client.py, helpers.py
- Created service file: auth_service.py (COMPLETE)

## Remaining Work

### 1. Service Files to Create

#### services/muinmos_service.py
Functions to move:
- get_muinmos_token()
- create_assessment()
- muinmos_assessment_check()
- check_muinmos_assessment_to_send_kycpdf()
- update_order_assessment_iscomplete_sendpdfreport()
- get_muinmos_question()
- submit_muinmos_answer()
- create_muinmos_assessment_by_guest_account()

#### services/guest_service.py
Functions to move:
- get_guest_account_profile()
- update_guest_account_profile()
- update_guest_account_notification_settings()
- get_order_payments_by_guest_account()
- get_search_histories_by_guest_account_id()
- get_service_info_by_order_code()

#### services/search_service.py
Functions to move:
- create_search_history()

#### services/stripe_service.py
Functions to move:
- _process_stripe_webhook_event()

#### services/checkout_service.py
Functions to move:
- checkout_start()

#### services/order_service.py
Functions to move:
- create_order()
- list_orders()
- get_order()
- get_order_payment_by_code()

#### services/service_service.py
Functions to move:
- list_service_prices()
- get_service_price()
- list_countries()

### 2. Router Files to Create

#### routers/auth.py
Endpoints:
- POST /auth/email-get-otp
- POST /auth/submit-otp
- POST /auth/validate-by-token-and-guest-account-id

#### routers/muinmos.py
Endpoints:
- GET /muinmos/token
- POST /muinmos/create-assessment
- POST /muinmos/assessment-check
- GET /muinmos/question/{assessment_id}
- POST /muinmos/submit-answer
- POST /muinmos/create-assessment-by-guest-account

#### routers/guest_account.py
Endpoints:
- POST /guest-account/profile
- POST /guest-account/update-profile
- POST /guest-account/update-notification-settings
- POST /guest-account/order-payments
- POST /guest-account/search-histories

#### routers/orders.py
Endpoints:
- POST /orders
- GET /orders
- GET /orders/{order_id}
- GET /order-payments/by-code/{order_code}

#### routers/checkout.py
Endpoints:
- POST /checkout/start

#### routers/services.py
Endpoints:
- GET /service-prices
- GET /service-prices/{service_price_id}
- GET /countries
- GET /service-info/{order_code}

### 3. Updated main.py Structure

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from db import Base, engine, SessionLocal
from utils.helpers import load_env_files
import os

# Import routers
from routers import auth, muinmos, guest_account, orders, checkout, services

app = FastAPI(title="KYC Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[...],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(muinmos.router)
app.include_router(guest_account.router)
app.include_router(orders.router)
app.include_router(checkout.router)
app.include_router(services.router)

# Basic endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to KYC Backend API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Startup event
@app.on_event("startup")
def on_startup():
    load_env_files()
    # Database migrations...

# Lambda handler
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except Exception:
    pass
```

## Migration Strategy

### Phase 1: Services (Week 1)
1. Create all service files
2. Move business logic from main.py to services
3. Test each service independently

### Phase 2: Routers (Week 2)
1. Create all router files
2. Move endpoint definitions from main.py to routers
3. Import services in routers
4. Test each router independently

### Phase 3: Integration (Week 3)
1. Update main.py to import and include all routers
2. Remove old code from main.py
3. Full integration testing
4. Update lambda_function.py if needed

### Phase 4: Cleanup (Week 4)
1. Remove unused imports
2. Add docstrings
3. Update README.md
4. Create API documentation

## Benefits After Refactoring

1. **File Sizes**: Each file will be 100-300 lines instead of 2000+
2. **Maintainability**: Easy to find and modify specific functionality
3. **Testing**: Can test individual modules
4. **Collaboration**: Multiple developers can work simultaneously
5. **Scalability**: Easy to add new features

## Example Router File

```python
# routers/auth.py
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from schemas.auth_schemas import LoginOTPRequest, SubmitOTPRequest, AuthValidationRequest
from services.auth_service import (
    login_with_email_generate_otp,
    login_submit_otp,
    auth_validation_by_token_and_guest_account_id
)
from db import SessionLocal

router = APIRouter(prefix="/auth", tags=["authentication"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/email-get-otp")
def login_with_email_generate_otp_endpoint(payload: LoginOTPRequest, db: Session = Depends(get_db)):
    return login_with_email_generate_otp(payload.email, payload.is_from_login, db)

@router.post("/submit-otp")
def login_submit_otp_endpoint(payload: SubmitOTPRequest, db: Session = Depends(get_db)):
    return login_submit_otp(payload.email, payload.otp, db)

@router.post("/validate-by-token-and-guest-account-id")
def auth_validation_endpoint(request: Request, payload: AuthValidationRequest, db: Session = Depends(get_db)):
    guest_account_token = request.headers.get("GuestAccountToken", "")
    return auth_validation_by_token_and_guest_account_id(payload.guest_account_id, guest_account_token, db)
```

## Next Steps

1. Review this guide
2. Decide on migration timeline
3. Start with Phase 1 (Services)
4. Test thoroughly after each phase
5. Deploy incrementally

## Notes

- Keep main.py as the entry point
- All business logic goes in services/
- All API endpoints go in routers/
- All request/response models go in schemas/
- All utilities go in utils/
- Database models stay in models.py
- Database connection stays in db.py
