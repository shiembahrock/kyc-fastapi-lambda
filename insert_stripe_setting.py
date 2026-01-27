import os
from db import SessionLocal
from models import StripeSetting

def main():
    secret = os.getenv("STRIPE_SECRET_KEY")
    if not secret:
        raise SystemExit("STRIPE_SECRET_KEY is not set")
    db = SessionLocal()
    obj = StripeSetting(secret_key=secret)
    db.add(obj)
    db.commit()
    print({"inserted": True, "stripe_setting_id": str(obj.stripe_setting_id)})
    db.close()

if __name__ == "__main__":
    main()
