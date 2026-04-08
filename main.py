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
        if env in ("local", "dev") and engine.dialect.name == "postgresql":
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
                
                q = text("SELECT character_maximum_length FROM information_schema.columns WHERE table_name='guest_login_sessions' AND column_name='token'")
                res = conn.execute(q).scalar()
                if res and res < 500:
                    conn.execute(text("ALTER TABLE guest_login_sessions ALTER COLUMN token TYPE VARCHAR(500)"))
                    conn.commit()
    except Exception:
        pass

# Lambda handler
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except Exception:
    pass