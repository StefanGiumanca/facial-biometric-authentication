from fastapi import FastAPI

from backend.api.routes_admin import router as admin_router
from backend.api.routes_kyc import router as kyc_router
from backend.db.database import init_db

app = FastAPI(title="Facial Biometric Authentication API")


@app.on_event("startup")
def startup_event():
    init_db()


app.include_router(kyc_router)
app.include_router(admin_router)
