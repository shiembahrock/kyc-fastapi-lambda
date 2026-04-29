from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from db import Base, engine, SessionLocal
from utils.helpers import load_env_files
import os

# Import all routers
from routers import auth, muinmos, guest_account, orders, stripe, services, admin

app = FastAPI(title="KYC Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://staging.d27qd8fx6l7txq.amplifyapp.com"
    ],
    allow_origin_regex=r"http://(localhost|127\\.0\\.0\\.1):\\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth.router)
app.include_router(muinmos.router)
app.include_router(guest_account.router)
app.include_router(orders.router)
app.include_router(stripe.router)
app.include_router(services.router)
app.include_router(admin.router)

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

# Startup event with migrations
@app.on_event("startup")
def on_startup():
    load_env_files()
    env = os.getenv("APP_ENV", "local").lower()
    if env in ("local", "dev"):
        Base.metadata.create_all(bind=engine)
    try:
        if engine.dialect.name == "postgresql":
            with engine.connect() as conn:
                # Database migrations...
                q = text("SELECT data_type FROM information_schema.columns WHERE table_name='orders' AND column_name='order_id'")
                res = conn.execute(q).scalar()
                if res and res.lower() != "uuid":
                    conn.execute(text("ALTER TABLE orders ALTER COLUMN order_id TYPE uuid USING order_id::uuid"))
                    conn.commit()

                q = text("SELECT column_name FROM information_schema.columns WHERE table_name='muinmos_settings' AND column_name='base_api_url'")
                res = conn.execute(q).scalar()
                if not res:
                    conn.execute(text("ALTER TABLE muinmos_settings ADD COLUMN base_api_url VARCHAR NOT NULL DEFAULT ''"))
                    conn.commit()

                q = text("SELECT column_name FROM information_schema.columns WHERE table_name='service_prices' AND column_name='is_popular'")
                res = conn.execute(q).scalar()
                if not res:
                    conn.execute(text("ALTER TABLE service_prices ADD COLUMN is_popular BOOLEAN NOT NULL DEFAULT false"))
                    conn.commit()

                q = text("SELECT column_name FROM information_schema.columns WHERE table_name='service_prices' AND column_name='kyc_profile_id'")
                res = conn.execute(q).scalar()
                if not res:
                    conn.execute(text("ALTER TABLE service_prices ADD COLUMN kyc_profile_id VARCHAR(50)"))
                    conn.commit()

                q = text("SELECT column_name FROM information_schema.columns WHERE table_name='service_prices' AND column_name='stripe_product_id'")
                res = conn.execute(q).scalar()
                if not res:
                    conn.execute(text("ALTER TABLE service_prices ADD COLUMN stripe_product_id VARCHAR(50)"))
                    conn.commit()

                q = text("SELECT character_maximum_length FROM information_schema.columns WHERE table_name='guest_login_sessions' AND column_name='token'")
                res = conn.execute(q).scalar()
                if res and res < 500:
                    conn.execute(text("ALTER TABLE guest_login_sessions ALTER COLUMN token TYPE VARCHAR(500)"))
                    conn.commit()
                
                # Seed CreditSourceType default rows
                seed_rows = [
                    ("Sign Up Referral",        "New User registered within Referral Code.",                                                          1,  "2.00", True,  True),
                    ("New Referral",            "Existing User applied Referral Code.",                                                               1,  "1.00", True,  True),
                    ("Transaction by Referral", "New transaction by Referral Member.",                                                                5,  "3.00", True,  True),
                    ("Use for Transaction",     "Debit transaction used by Credit Owner for transaction. Note : max_credit_count = -1 means infinity.", -1, "0.00", False, True),
                ]
                # Ensure source_type_id column is SERIAL (auto-increment) in production
                conn.execute(text("CREATE SEQUENCE IF NOT EXISTS credit_source_type_seq"))
                conn.execute(text("ALTER TABLE credit_source_types ALTER COLUMN source_type_id SET DEFAULT nextval('credit_source_type_seq')"))
                conn.commit()
                for name, description, max_count, amount, is_credit, is_activated in seed_rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM credit_source_types WHERE source_type_name = :n"),
                        {"n": name}
                    ).scalar()
                    if not exists:
                        conn.execute(
                            text("""
                                INSERT INTO credit_source_types
                                (credit_source_type_id, source_type_name, source_type_description, max_credit_count, credit_amount,
                                 created_at, start_date, end_date, is_credit, is_activated)
                                VALUES (gen_random_uuid(), :n, :desc, :mc, :ca, now(), now(),
                                        now() + interval '50 years', :ic, :ia)
                            """),
                            {"n": name, "desc": description, "mc": max_count, "ca": amount, "ic": is_credit, "ia": is_activated}
                        )
                        conn.commit()
    except Exception:
        pass

# Lambda handler
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except Exception:
    pass