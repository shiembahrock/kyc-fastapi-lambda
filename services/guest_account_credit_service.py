import os
import hmac
import hashlib
from decimal import Decimal
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import GuestAccountCredit, CreditSourceType
from datetime import timezone

def generate_credit_balance_signature(balance_amount: Decimal, guest_account_id, created_at: datetime) -> str:
    secret = os.getenv("CREDIT_BALANCE_SECRET", "")
    message = f"{balance_amount}{guest_account_id}{created_at.isoformat()}"
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

def is_valid_credit_balance_signature(guest_account_id, db: Session) -> bool:
    lgac = db.query(GuestAccountCredit).filter(
        GuestAccountCredit.guest_account_id == guest_account_id
    ).order_by(GuestAccountCredit.created_at.desc()).first()
    if not lgac:
        return True
    return lgac.balance_signature == generate_credit_balance_signature(lgac.balance_amount, lgac.guest_account_id, lgac.created_at)

def credit_balance_correction(guest_account_id, db: Session):
    result = db.query(
        func.sum(GuestAccountCredit.credit_amount).label("CA"),
        func.sum(GuestAccountCredit.debit_amount).label("DA")
    ).filter(GuestAccountCredit.guest_account_id == guest_account_id).one()

    balance_amount = (result.CA or Decimal("0")) - (result.DA or Decimal("0"))

    lgac = db.query(GuestAccountCredit).filter(
        GuestAccountCredit.guest_account_id == guest_account_id
    ).order_by(GuestAccountCredit.created_at.desc()).first()

    if not lgac:
        return

    balance_signature = generate_credit_balance_signature(balance_amount, guest_account_id, lgac.created_at)
    lgac.balance_amount = balance_amount
    lgac.balance_signature = balance_signature
    db.commit()

def insert_guest_account_credit_transaction(guest_account_id, reference_id, source_type_id: int, db: Session, debit_amount: Decimal = Decimal("0.00")) -> bool:
    if not is_valid_credit_balance_signature(guest_account_id, db):
        credit_balance_correction(guest_account_id, db)

    created_at = datetime.now(timezone.utc)

    cst = db.query(CreditSourceType).filter(
        CreditSourceType.source_type_id == source_type_id,
        CreditSourceType.start_date <= created_at,
        CreditSourceType.end_date >= created_at,
        CreditSourceType.is_activated == True
    ).first()
    if not cst:
        return False

    lgac = db.query(GuestAccountCredit).filter(
        GuestAccountCredit.guest_account_id == guest_account_id
    ).order_by(GuestAccountCredit.created_at.desc()).first()

    prev_balance = lgac.balance_amount if lgac else Decimal("0.00")

    if source_type_id in (1, 2, 3):
        gac_count = db.query(func.count(GuestAccountCredit.guest_account_credit_id)).filter(
            GuestAccountCredit.guest_account_id == guest_account_id,
            GuestAccountCredit.reference_id == reference_id
        ).scalar()
        if cst.max_credit_count <= gac_count:
            return False
        balance_amount = prev_balance + cst.credit_amount
        cbs = generate_credit_balance_signature(balance_amount, guest_account_id, created_at)
        db.add(GuestAccountCredit(
            guest_account_id=guest_account_id,
            reference_id=reference_id,
            created_at=created_at,
            credit_source_type_id=cst.credit_source_type_id,
            credit_amount=cst.credit_amount,
            debit_amount=debit_amount,
            balance_amount=balance_amount,
            balance_signature=cbs
        ))
        db.commit()
    elif source_type_id == 4:
        balance_amount = prev_balance - debit_amount
        cbs = generate_credit_balance_signature(balance_amount, guest_account_id, created_at)
        db.add(GuestAccountCredit(
            guest_account_id=guest_account_id,
            reference_id=reference_id,
            created_at=created_at,
            credit_source_type_id=cst.credit_source_type_id,
            credit_amount=Decimal("0.00"),
            debit_amount=debit_amount,
            balance_amount=balance_amount,
            balance_signature=cbs
        ))
        db.commit()

    return True
