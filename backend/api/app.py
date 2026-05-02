from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes_kyc import router as kyc_router
from backend.api.routes_admin import router as admin_router
from backend.db.database import init_db

app = FastAPI(title="Facial Biometric Authentication API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(kyc_router)
app.include_router(admin_router)
