# main.py - FINAL CLEAN VERSION
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import motor.motor_asyncio
import os
import json
import time
import requests
import logging
import openai
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="🚀 Telnyx SMS Agent API", 
    description="AI-powered SMS messaging with Telnyx integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telnyx_main")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGO_DB", "test")]

# Telnyx Configuration
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
TELNYX_PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER", "+15153495568")
TELNYX_MESSAGING_PROFILE_ID = os.getenv("TELNYX_MESSAGING_PROFILE_ID", "400198ac-d6df-446c-81c7-1bd81885937f")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Active conversations tracking
active_conversations: Dict[str, dict] = {}

class AgentConfig(BaseModel):
    instructions: str
    welcome_message: str
    voice: str = "coral"

class DirectSMSProcessor:
    """Process SMS messages with OpenAI integration"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
    async def get_current_config(self):
        """Get current agent configuration from database"""
        try:
            config = await db.agent_configs.find_one(
                {"active": True},
                sort=[("updated_at", -1)]
            )
            
            if config:
                return {
                    "instructions": config["instructions"],
                    "welcome_message": config["welcome_message"], 
                    "voice": config["voice"]
                }
            else:
                return {
                    "instructions": "You are a helpful AI assistant.",
                    "welcome_message": "Hello! How can I help you today?",
                    "voice": "coral"
                }
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            return {
                "instructions": "You are a helpful AI assistant.",
                "welcome_message": "Hello! How can I help you today?",
                "voice": "coral"
            }
    
    def format_for_sms(self, text: str) -> str:
        """Format text for SMS delivery"""
        if not text:
            return ""
        
        formatted = text.strip()
        
        # Remove markdown formatting
        formatted = formatted.replace("**", "").replace("*", "")
        formatted = formatted.replace("```", "").replace("`", "")
        
        # SMS length limit
        if len(formatted) > 1600:
            formatted = formatted[:1597] + "..."
        
        return formatted
    
    async def process_message(self, user_message: str) -> str:
        """Process user message and generate AI response"""
        try:
            if not self.openai_client:
                return "AI service is not available right now."
            
            # Get current config
            config = await self.get_current_config()
            instructions = config.get("instructions", "You are a helpful assistant.")
            
            # Create messages for OpenAI
            messages = [
                {
                    "role": "system", 
                    "content": f"{instructions}\n\nRespond naturally and keep responses concise for SMS messaging (under 1600 characters)."
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ]
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Format for SMS
            formatted_response = self.format_for_sms(ai_response)
            
            logger.info(f"✅ Generated AI response: {formatted_response[:100]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Error processing message with AI: {e}")
            return "Sorry, I'm having trouble processing your message right now. Please try again."

# Initialize SMS processor
sms_processor = DirectSMSProcessor()

def get_sender_for_country(phone_number: str) -> str:
    """Get appropriate sender based on destination country"""
    clean_phone = ''.join(c for c in phone_number if c.isdigit() or c == '+')
    
    # For now, let's use your phone number for ALL destinations
    # This should work for most countries including Pakistan
    return TELNYX_PHONE_NUMBER
    
    # US/Canada numbers can use phone number
    # if clean_phone.startswith('+1'):
    #     return TELNYX_PHONE_NUMBER
    
    # # Pakistan - try common approved senders
    # if clean_phone.startswith('+92'):
    #     # Try these common approved alpha senders for Pakistan
    #     return 'INFO'  # or try 'ALERT', 'NOTIFY', 'SMS'
    
    # # International numbers need alpha sender
    # country_senders = {
    #     '+91': 'INFO',    # India  
    #     '+44': 'INFO',    # UK
    #     '+33': 'INFO',    # France
    #     '+49': 'INFO',    # Germany
    #     '+86': 'INFO',    # China
    # }
    
    # for country_code, sender in country_senders.items():
    #     if clean_phone.startswith(country_code):
    #         return sender
    
    # # Default to INFO for any international number
    # return 'INFO'

def send_sms_direct(phone_number: str, message: str) -> dict:
    """Direct SMS sending function"""
    try:
        # Clean phone number
        clean_phone = ''.join(c for c in phone_number if c.isdigit() or c == '+')
        if not clean_phone.startswith('+'):
            clean_phone = '+' + clean_phone
        
        # Get appropriate sender
        sender = get_sender_for_country(clean_phone)
        
        # Telnyx API call
        url = "https://api.telnyx.com/v2/messages"
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "from": sender,
            "to": clean_phone,
            "text": message
        }
        
        # Don't use messaging profile for phone number senders
        # Add messaging profile for alpha senders (international)
        # if sender != TELNYX_PHONE_NUMBER and TELNYX_MESSAGING_PROFILE_ID:
        #     payload["messaging_profile_id"] = TELNYX_MESSAGING_PROFILE_ID
        #     logger.info(f"📋 Using messaging profile for alpha sender: {sender}")
        
        logger.info(f"📋 Sending with phone number (no profile needed): {sender}")
        
        logger.info(f"📱 Sending to {clean_phone} from {sender}: {message}")
        logger.info(f"📦 Payload: {payload}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        logger.info(f"📊 Status: {response.status_code}")
        logger.info(f"📄 Response: {response.text}")
        
        if response.status_code in [200, 201]:
            result = response.json()
            return {
                "status": "✅ SUCCESS",
                "phone": clean_phone,
                "message": message,
                "sender": sender,
                "message_id": result.get('data', {}).get('id'),
                "success": True
            }
        else:
            return {
                "status": "❌ FAILED",
                "phone": clean_phone,
                "sender": sender,
                "error": response.text,
                "status_code": response.status_code,
                "success": False
            }
            
    except Exception as e:
        logger.error(f"❌ SMS Error: {e}")
        return {
            "status": "❌ ERROR",
            "phone": phone_number,
            "error": str(e),
            "success": False
        }

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE SMS ENDPOINT - HANDLES EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/send")
async def send_sms(phone: str, message: str):
    """
    🚀 UNIVERSAL SMS SENDER
    
    Automatically handles:
    - Multiple countries/numbers
    - Appropriate sender selection  
    - AI processing (if connected)
    - Database logging
    
    Usage: /send?phone=+923117762632&message=Hello
    """
    try:
        # Send SMS
        result = send_sms_direct(phone, message)
        
        # Save to database if successful
        if result.get("success"):
            try:
                await db.messages.insert_one({
                    "message_id": result.get("message_id"),
                    "from_sender": result.get("sender"),
                    "to_number": result.get("phone"),
                    "message_text": message,
                    "direction": "outbound",
                    "timestamp": datetime.utcnow(),
                    "event_type": "universal_send",
                    "status": "sent"
                })
                logger.info("✅ Message saved to database")
            except Exception as db_error:
                logger.warning(f"⚠️ Database save failed: {db_error}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Send SMS Error: {e}")
        return {
            "status": "❌ ERROR",
            "phone": phone,
            "message": message,
            "error": str(e),
            "success": False
        }

# ═══════════════════════════════════════════════════════════════════════════════
# AI INTEGRATION FOR OPENAIAGENT.PY
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process-and-send")
async def process_and_send_sms(request: Request):
    """
    🤖 AI PROCESSING + SMS SENDING
    
    For integration with openaiagent.py
    Processes incoming message with AI and sends response
    
    POST Body: {
        "from_number": "+923117762632",
        "message": "Hello, I need help"
    }
    """
    try:
        data = await request.json()
        from_number = data.get("from_number")
        incoming_message = data.get("message")
        
        if not from_number or not incoming_message:
            raise HTTPException(400, "Missing from_number or message")
        
        logger.info(f"🤖 AI Processing from {from_number}: {incoming_message}")
        
        # Process with AI
        ai_response = await sms_processor.process_message(incoming_message)
        
        # Send AI response back
        sms_result = send_sms_direct(from_number, ai_response)
        
        # Save both messages to database
        try:
            # Save incoming message
            await db.messages.insert_one({
                "from_number": from_number,
                "to_number": TELNYX_PHONE_NUMBER,
                "message_text": incoming_message,
                "direction": "inbound",
                "timestamp": datetime.utcnow(),
                "event_type": "ai_processing"
            })
            
            # Save AI response
            if sms_result.get("success"):
                await db.messages.insert_one({
                    "message_id": sms_result.get("message_id"),
                    "from_sender": sms_result.get("sender"),
                    "to_number": from_number,
                    "message_text": ai_response,
                    "direction": "outbound",
                    "timestamp": datetime.utcnow(),
                    "event_type": "ai_response"
                })
        except Exception as db_error:
            logger.warning(f"⚠️ Database save failed: {db_error}")
        
        return {
            "status": "success",
            "from_number": from_number,
            "incoming_message": incoming_message,
            "ai_response": ai_response,
            "sms_sent": sms_result.get("success"),
            "sms_result": sms_result
        }
        
    except Exception as e:
        logger.error(f"❌ Process and send error: {e}")
        raise HTTPException(500, str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/update-agent")
async def update_agent(config: AgentConfig):
    """Update agent configuration"""
    try:
        await db.agent_configs.update_many({}, {"$set": {"active": False}})
        
        result = await db.agent_configs.insert_one({
            "instructions": config.instructions,
            "welcome_message": config.welcome_message,
            "voice": config.voice,
            "updated_at": datetime.utcnow(),
            "active": True
        })
        
        logger.info(f"✅ Agent config updated: {result.inserted_id}")
        return {
            "status": "success", 
            "message": "Agent updated successfully",
            "config_id": str(result.inserted_id)
        }
    
    except Exception as e:
        logger.error(f"❌ Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-agent")
async def get_current_agent():
    """Get current active agent configuration"""
    try:
        config = await db.agent_configs.find_one(
            {"active": True},
            sort=[("updated_at", -1)]
        )
        
        if config:
            return {
                "instructions": config["instructions"],
                "welcome_message": config["welcome_message"], 
                "voice": config["voice"],
                "config_id": str(config["_id"])
            }
        else:
            return {
                "instructions": "You are a helpful AI assistant.",
                "welcome_message": "Hello! How can I help you today?",
                "voice": "coral"
            }
            
    except Exception as e:
        logger.error(f"❌ Error getting agent config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_conversations():
    """Get recent conversations from database"""
    try:
        # Get last 50 messages
        messages = await db.messages.find().sort("timestamp", -1).limit(50).to_list(50)
        
        return {
            "messages": [
                {
                    "id": str(msg.get("_id")),
                    "from": msg.get("from_number") or msg.get("from_sender"),
                    "to": msg.get("to_number"),
                    "text": msg.get("message_text"),
                    "direction": msg.get("direction"),
                    "timestamp": msg.get("timestamp"),
                    "status": msg.get("status", "unknown")
                }
                for msg in messages
            ],
            "total": len(messages)
        }
    except Exception as e:
        logger.error(f"❌ Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await db.command("ping")
        
        return {
            "status": "healthy",
            "database": "connected",
            "telnyx": "configured" if TELNYX_API_KEY else "not_configured",
            "openai": "configured" if OPENAI_API_KEY else "not_configured",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 Telnyx SMS Agent API with AI Processing",
        "version": "2.0.0",
        "primary_endpoint": "/send?phone=+XXXXXXXXXXX&message=Hello",
        "ai_endpoint": "/api/process-and-send (POST)",
        "features": [
            "Universal SMS sending to any country",
            "Automatic sender selection",
            "AI-powered responses",
            "Database logging",
            "Multiple number support"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Telnyx SMS Agent API on port 7000...")
    uvicorn.run(app, host="0.0.0.0", port=7000)