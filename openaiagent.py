import os
import time
import json
import asyncio
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict

# Simple fallback for LlamaIndex (RAG not critical for SMS)
LLAMA_INDEX_AVAILABLE = False
print("ℹ️ Running in SMS-only mode (no RAG)")

class Document:
    def __init__(self, text="", metadata=None):
        self.text = text
        self.metadata = metadata or {}

class VectorStoreIndex:
    @classmethod
    def from_documents(cls, documents):
        return cls()
    
    def as_retriever(self):
        return DummyRetriever()

class DummyRetriever:
    async def aretrieve(self, query):
        return []

from livekit import agents, api
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli
)
from livekit.plugins import openai

# ─── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# Configuration
MAIN_API_URL = "http://localhost:7000"  # main.py API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
agent_logger = logging.getLogger("sms_agent")

# ─── Configuration Manager ─────────────────────────────────────────────────────
class ConfigManager:
    """Manage agent configuration from main.py API"""
    
    def __init__(self, api_url: str = MAIN_API_URL):
        self.api_url = api_url.rstrip("/")
    
    def fetch_config(self, retries: int = 3, timeout: float = 10.0) -> dict:
        """Fetch agent configuration from main.py API"""
        url = f"{self.api_url}/api/get-agent"
        last_exc = None
        
        for attempt in range(1, retries + 1):
            try:
                agent_logger.info(f"Fetching config from {url} (attempt {attempt})")
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                config = resp.json()
                agent_logger.info(f"✅ Config fetched successfully")
                return config
            except Exception as e:
                last_exc = e
                agent_logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    time.sleep(2)
        
        agent_logger.error(f"❌ All {retries} attempts failed. Using fallback config.")
        return self._get_fallback_config()
    
    def _get_fallback_config(self) -> dict:
        """Fallback configuration when API is unavailable"""
        return {
            "instructions": "You are a helpful AI assistant. Respond naturally and conversationally.",
            "welcome_message": "Hello! How can I help you today?",
            "voice": "coral"
        }

# ─── SMS Message Processor ─────────────────────────────────────────────────────
class SMSMessageProcessor:
    """Process messages for SMS delivery"""
    
    def __init__(self, config: dict):
        self.config = config
        self.openai_api_key = OPENAI_API_KEY
    
    async def process_message(self, user_message: str) -> str:
        """Process user message and generate response"""
        try:
            import openai as openai_client
            
            client = openai_client.OpenAI(api_key=self.openai_api_key)
            
            # Build prompt from config
            instructions = self.config.get("instructions", "You are a helpful assistant.")
            
            messages = [
                {"role": "system", "content": f"{instructions}\n\nRespond naturally and keep responses concise for SMS messaging."},
                {"role": "user", "content": user_message}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Format for SMS
            formatted_response = self._format_for_sms(ai_response)
            
            agent_logger.info(f"Generated response: {formatted_response[:100]}...")
            return formatted_response
            
        except Exception as e:
            agent_logger.error(f"Error processing message: {e}")
            return "Sorry, I'm having trouble processing your message right now. Please try again."
    
    def _format_for_sms(self, text: str) -> str:
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

# ─── Simple SMS Agent ──────────────────────────────────────────────────────────
class TelnyxSMSAgent(Agent):
    """Simple agent for SMS message processing"""
    
    def __init__(self, config: dict):
        instructions = config.get("instructions", "You are a helpful AI assistant.")
        super().__init__(instructions=instructions)
        self.config = config
        self.message_processor = SMSMessageProcessor(config)
        
    async def on_session_start(self, session: AgentSession):
        """Called when agent session starts"""
        agent_logger.info("📱 SMS Agent session started")
        
    async def process_sms(self, message: str) -> str:
        """Process SMS message and return response"""
        return await self.message_processor.process_message(message)

# ─── Main Entrypoint ───────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the SMS agent"""
    agent_logger.info("📱 Starting SMS Message Processing Agent...")
    
    # 1) Connect to room (minimal setup for messaging)
    await ctx.connect()
    agent_logger.info("✅ Connected to LiveKit room")
    
    # 2) Initialize config manager
    config_manager = ConfigManager()
    
    # 3) Fetch configuration
    config = config_manager.fetch_config()
    agent_logger.info(f"Using config for SMS processing")
    
    # 4) Create SMS agent
    agent = TelnyxSMSAgent(config)
    
    # 5) Create simple session with OpenAI LLM
    session = AgentSession(
        llm=openai.LLM(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY
        )
    )
    
    # 6) Start session (minimal setup for SMS)
    await session.start(
        room=ctx.room,
        agent=agent
    )
    
    # 7) Send welcome message if this is a new conversation
    welcome_message = config.get("welcome_message", "Hello! How can I help you today?")
    
    # For SMS, we'll just log the welcome message
    agent_logger.info(f"📱 SMS Agent ready! Welcome message: {welcome_message}")
    
    # 8) Keep agent alive for message processing
    agent_logger.info("🎉 SMS Agent ready for message processing!")
    
    # Simple message processing loop
    while True:
        try:
            await asyncio.sleep(1)  # Keep alive
            # In real implementation, this would process incoming messages
            # from webhooks or message queues
        except Exception as e:
            agent_logger.error(f"Error in message loop: {e}")
            await asyncio.sleep(5)

# ─── CLI Launcher ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent_logger.info("📱 Launching SMS Processing Agent with LiveKit...")
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="telnyx-sms-agent"
        )
    )
if __name__ == "__main__":
    agent_logger.info("🚀 Launching OpenAI Agent with LiveKit CLI...")
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="telnyx-openai-agent"
        )
    )