from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from db import SessionLocal
from schemas.muinmos_schemas import SubmitMuinmosAnswerRequest, CreateMuinmosAssessmentByGuestAccountRequest
from services.muinmos_service import (
    get_muinmos_token,
    create_assessment,
    muinmos_assessment_check,
    get_muinmos_question,
    submit_muinmos_answer,
    create_muinmos_assessment_by_guest_account
)

router = APIRouter(prefix="/muinmos", tags=["muinmos"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/token")
def get_muinmos_token_endpoint(db: Session = Depends(get_db)):
    return get_muinmos_token(db)

@router.post("/create-assessment")
def create_assessment_endpoint(user_email: str, order_code: str, db: Session = Depends(get_db)):
    return create_assessment(user_email, order_code, db)

@router.post("/assessment-check")
def muinmos_assessment_check_endpoint(db: Session = Depends(get_db)):
    return muinmos_assessment_check(db)

@router.get("/question/{assessment_id}")
def get_muinmos_question_endpoint(assessment_id: str, db: Session = Depends(get_db)):
    return get_muinmos_question(assessment_id, db)

@router.post("/submit-answer")
def submit_muinmos_answer_endpoint(request: Request, payload: SubmitMuinmosAnswerRequest, db: Session = Depends(get_db)):
    guest_account_id = request.headers.get("GuestAccountId", "")
    guest_login_token = request.headers.get("GuestLoginToken", "")
    return submit_muinmos_answer(
        guest_account_id,
        guest_login_token,
        payload.assessment_id,
        payload.answer,
        db
    )

@router.post("/create-assessment-by-guest-account")
def create_muinmos_assessment_by_guest_account_endpoint(request: Request, payload: CreateMuinmosAssessmentByGuestAccountRequest, db: Session = Depends(get_db)):
    guest_account_id = request.headers.get("GuestAccountId", "")
    guest_login_token = request.headers.get("GuestLoginToken", "")
    return create_muinmos_assessment_by_guest_account(
        guest_account_id,
        guest_login_token,
        payload.order_code,
        db
    )
