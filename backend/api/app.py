from fastapi import FastAPI
from backend.api.routes_kyc import router as kyc_router

app = FastAPI(title="Facial Biometric Authentication API")

app.include_router(kyc_router)