# api.py

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId

# ─── Load env & validate ───────────────────────────────────────────────────────
load_dotenv()
for var in ("MONGO_URI", "MONGO_DB", "MONGO_COLLECTION"):
    if not os.getenv(var):
        raise RuntimeError(f"Missing required .env variable: {var!r}")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger("api")

# ─── Singleton Mongo client & collection ───────────────────────────────────────
_mongo_client     = None
_mongo_collection = None

def get_mongo_collection():
    """
    Returns the pre-initialized collection handle.
    Does NOT ping or recreate the client.
    """
    global _mongo_collection
    if _mongo_collection is None:
        raise RuntimeError("Mongo not initialized; did startup_event run?")
    return _mongo_collection

# ─── Lifespan event handler ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events using the new lifespan context manager.
    """
    global _mongo_client, _mongo_collection

    # Startup
    uri       = os.getenv("MONGO_URI")
    db_name   = os.getenv("MONGO_DB")
    coll_name = os.getenv("MONGO_COLLECTION")

    logger.info(f"Starting up: connecting to MongoDB {db_name}.{coll_name}…")
    _mongo_client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        maxPoolSize=50,
        retryWrites=True
    )

    try:
        _mongo_client.admin.command("ping")
        logger.info("✅ MongoDB ping successful.")
    except Exception as e:
        logger.error(f"❌ MongoDB ping failed at startup: {e!r}")
        raise

    _mongo_collection = _mongo_client[db_name][coll_name]
    
    yield  # App runs here
    
    # Shutdown
    if _mongo_client:
        logger.info("Shutting down: closing MongoDB connection.")
        _mongo_client.close()

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="Dynamic Prompt Config API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ─── Request model ─────────────────────────────────────────────────────────────
class DocumentRequest(BaseModel):
    documentId: str

class TelnyxConfig(BaseModel):
    phone_number: str
    trunk_name: str = "Telnyx Outbound Trunk"
    sip_address: str = "sip.telnyx.com"

# ─── POST /document ────────────────────────────────────────────────────────────
@app.post("/document")
async def set_document(req: DocumentRequest):
    logger.info(f"▶️  POST /document payload: {req.documentId!r}")

    if not ObjectId.is_valid(req.documentId):
        logger.warning("⚠️  Invalid ObjectId format.")
        raise HTTPException(400, "Invalid documentId format.")

    coll = get_mongo_collection()
    oid  = ObjectId(req.documentId)

    if not coll.find_one({"_id": oid}):
        logger.warning(f"⚠️  No document found with _id={req.documentId}")
        raise HTTPException(404, "No config found for this documentId.")

    off = coll.update_many({"active": True}, {"$set": {"active": False}})
    logger.info(f"🔄 Deactivated {off.modified_count} previously active docs.")

    on = coll.update_one({"_id": oid}, {"$set": {"active": True}})
    if on.modified_count != 1:
        logger.error("❌  Failed to activate the chosen document.")
        raise HTTPException(500, "Could not activate the document.")

    logger.info(f"✅  Document {req.documentId} is now active.")
    return {"status": "success", "documentId": req.documentId}


# ─── GET /config ──────────────────────────────────────────────────────────────
@app.get("/config")
async def get_current_config():
    logger.info("▶️  GET /config called.")

    coll = get_mongo_collection()
    doc  = coll.find_one({"active": True})

    if not doc:
        logger.warning("⚠️  No active config set in database.")
        raise HTTPException(404, "No active config set.")

    logger.info(
        f"✅  Returning config for documentId={doc['_id']} "
        f"(instructions length={len(doc.get('coreInstructions',''))}, "
        f"voice={doc.get('voice','<none>')}, "
        f"content length={len(doc.get('documentText', ''))})"
    )

    return {
        "documentId":       str(doc["_id"]),
        "filename":         doc.get("filename", ""),
        "content":          doc.get("documentText", ""),  # RAG-relevant full text
        "coreInstructions": doc.get("coreInstructions", ""),
        "welcomeMessage":   doc.get("welcomeMessage", ""),
        "voice":            doc.get("voice", ""),
        "telnyxConfig":     doc.get("telnyxConfig", {
            "phone_number": "+15153495568",
            "trunk_name": "Telnyx Outbound Trunk",
            "sip_address": "sip.telnyx.com",
            "auth_username": "livekituser123",
            "auth_password": "secret123!"
        })
    }

# ─── POST /telnyx-config ──────────────────────────────────────────────────────
@app.post("/telnyx-config")
async def update_telnyx_config(config: TelnyxConfig):
    logger.info(f"▶️  POST /telnyx-config: {config.phone_number}")
    
    coll = get_mongo_collection()
    doc = coll.find_one({"active": True})
    
    if not doc:
        logger.warning("⚠️  No active config to update.")
        raise HTTPException(404, "No active config found.")
    
    # Update the active document with Telnyx config
    result = coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "telnyxConfig": {
                "phone_number": config.phone_number,
                "trunk_name": config.trunk_name,
                "sip_address": config.sip_address,
                "auth_username": "livekituser123",
                "auth_password": "secret123!"
            }
        }}
    )
    
    if result.modified_count == 1:
        logger.info(f"✅  Telnyx config updated for document {doc['_id']}")
        return {"status": "success", "message": "Telnyx configuration updated"}
    else:
        logger.error("❌  Failed to update Telnyx config.")
        raise HTTPException(500, "Could not update Telnyx configuration.")

# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    logger.info("▶️  GET /health called.")
    try:
        # A simple ping
        get_mongo_collection().database.client.admin.command("ping")
        logger.info("✅ Health check: database connected.")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"❌ Health check failed: {e!r}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)