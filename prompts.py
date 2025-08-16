# prompts.py - SMS & Messaging Optimized

INSTRUCTIONS = """You are a helpful AI assistant. Provide clear, concise, and helpful responses. Keep your responses natural and conversational for text messaging and voice calls."""

WELCOME_MESSAGE = """Hello! How can I help you today?"""

VOICE = "coral"

# Telnyx Configuration for SMS/Voice
TELNYX_CONFIG = {
    "phone_number": "",
    "auth_username": "", 
    "auth_password": "",
    "sip_trunk_id": ""
}

def build_prompt(user_message: str, cfg: dict) -> str:
    """
    Build prompt using configuration from API
    cfg = {
      "welcome_message": "...",
      "instructions": "...", 
      "voice": "..."
    }
    """
    welcome = cfg.get('welcome_message', WELCOME_MESSAGE)
    instructions = cfg.get('instructions', INSTRUCTIONS)
    
    return (
        f"{welcome}\n\n"
        f"{instructions}\n\n"
        f"User's message: {user_message}\n\n"
        f"Respond naturally and helpfully."
    )

def format_for_sms(text: str) -> str:
    """
    Format text for SMS delivery
    """
    if not text:
        return ""
    
    # Clean up the text
    formatted = text.strip()
    
    # Remove markdown formatting that doesn't work in SMS
    formatted = formatted.replace("**", "").replace("*", "")
    formatted = formatted.replace("```", "").replace("`", "")
    formatted = formatted.replace("___", "").replace("__", "")
    
    # Remove extra newlines
    formatted = "\n".join(line.strip() for line in formatted.split("\n") if line.strip())
    
    # SMS length limit (160 chars for single SMS, 1600 for concatenated)
    if len(formatted) > 1600:
        formatted = formatted[:1597] + "..."
    
    return formatted

def format_for_voice(text: str) -> str:
    """
    Format text for voice delivery (calls)
    """
    if not text:
        return ""
    
    formatted = text.strip()
    
    # Remove markdown
    formatted = formatted.replace("**", "").replace("*", "")
    formatted = formatted.replace("```", "").replace("`", "")
    
    # Replace symbols with words for better voice synthesis
    replacements = {
        "&": "and",
        "@": "at",
        "#": "number", 
        "%": "percent",
        "$": "dollars",
        "+": "plus",
        "=": "equals",
        "<": "less than",
        ">": "greater than"
    }
    
    for symbol, word in replacements.items():
        formatted = formatted.replace(symbol, f" {word} ")
    
    # Clean up extra spaces
    formatted = " ".join(formatted.split())
    
    return formatted

def get_telnyx_config():
    """
    Get Telnyx configuration
    """
    return TELNYX_CONFIG

# Response templates for common scenarios
RESPONSE_TEMPLATES = {
    "greeting": "Hi! How can I help you today?",
    "confirmation": "Got it! Let me help you with that.",
    "error": "Sorry, I didn't understand. Could you clarify?", 
    "goodbye": "Thanks for reaching out! Have a great day!",
    "processing": "I'm working on that for you...",
    "help": "I'm here to help! What would you like to know?"
}

def get_response_template(scenario: str, custom_message: str = None) -> str:
    """
    Get a template response for common scenarios
    """
    if custom_message:
        return format_for_sms(custom_message)
    
    return RESPONSE_TEMPLATES.get(scenario, RESPONSE_TEMPLATES["help"])

def create_agent_prompt(user_input: str, context: str = "", conversation_history: list = None) -> str:
    """
    Create a comprehensive prompt for the AI agent
    
    Args:
        user_input: The user's message
        context: Additional context from RAG or other sources
        conversation_history: Previous conversation messages
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Base instructions
    prompt_parts.append(INSTRUCTIONS)
    
    # Add context if available
    if context:
        prompt_parts.append(f"Relevant information:\n{context}")
    
    # Add conversation history if available
    if conversation_history:
        history_text = "\n".join([
            f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
            for msg in conversation_history[-3:]  # Last 3 exchanges
        ])
        prompt_parts.append(f"Recent conversation:\n{history_text}")
    
    # Current user message
    prompt_parts.append(f"User's current message: {user_input}")
    
    # Instructions for response
    prompt_parts.append("Respond helpfully, naturally, and conversationally.")
    
    return "\n\n".join(prompt_parts)

# Telnyx SMS sending function
def send_telnyx_sms(to_number: str, message: str, from_number: str = None):
    """
    Send SMS via Telnyx API
    
    Args:
        to_number: Recipient phone number
        message: Message text
        from_number: Sender number (optional)
    
    Returns:
        API response or None if failed
    """
    import requests
    import os
    
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("Error: TELNYX_API_KEY not found")
        return None
    
    if not from_number:
        from_number = TELNYX_CONFIG["phone_number"]
    
    # Format phone numbers
    def format_phone(phone):
        cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
        return cleaned if cleaned.startswith('+') else '+' + cleaned
    
    to_number = format_phone(to_number)
    from_number = format_phone(from_number)
    
    # Format message
    formatted_message = format_for_sms(message)
    
    url = "https://api.telnyx.com/v2/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "from": from_number,
        "to": to_number,
        "text": formatted_message
    }
    
    # Add messaging profile if available
    messaging_profile_id = os.getenv("TELNYX_MESSAGING_PROFILE_ID")
    if messaging_profile_id:
        payload["messaging_profile_id"] = messaging_profile_id
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"SMS sent successfully to {to_number}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Failed to send SMS: {e}")
        return None

# Message validation and preprocessing
def validate_and_clean_message(message: str) -> tuple[bool, str]:
    """
    Validate and clean incoming message
    
    Returns:
        (is_valid, cleaned_message)
    """
    if not message or not message.strip():
        return False, "Empty message"
    
    cleaned = message.strip()
    
    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())
    
    # Check for maximum length
    if len(cleaned) > 5000:  # Reasonable limit
        return False, "Message too long"
    
    # Basic spam/abuse detection (you can enhance this)
    spam_indicators = ['spam', 'scam', 'urgent!!!', 'click here now']
    if any(indicator in cleaned.lower() for indicator in spam_indicators):
        return False, "Potential spam detected"
    
    return True, cleaned

def extract_phone_number(text: str) -> str:
    """
    Extract phone number from text
    """
    import re
    
    # Pattern for various phone number formats
    patterns = [
        r'\+\d{1,3}\s?\d{10,14}',  # +1 234567890
        r'\(\d{3}\)\s?\d{3}-\d{4}',  # (123) 456-7890
        r'\d{3}-\d{3}-\d{4}',  # 123-456-7890
        r'\d{10,11}'  # 1234567890
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    
    return ""

# Helper functions for better user experience
def is_question(text: str) -> bool:
    """Check if text is a question"""
    question_indicators = ['?', 'what', 'how', 'when', 'where', 'why', 'who', 'can', 'could', 'would', 'should']
    text_lower = text.lower()
    return '?' in text or any(text_lower.startswith(word) for word in question_indicators)

def get_response_tone(user_message: str) -> str:
    """Determine appropriate response tone based on user message"""
    urgent_words = ['urgent', 'emergency', 'help', 'asap', 'immediately']
    casual_words = ['hi', 'hello', 'hey', 'thanks', 'cool']
    
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in urgent_words):
        return "urgent"
    elif any(word in message_lower for word in casual_words):
        return "casual"
    else:
        return "professional"

# Analytics and logging helpers
def log_interaction(user_message: str, bot_response: str, metadata: dict = None):
    """Log interaction for analytics"""
    import logging
    import json
    from datetime import datetime
    
    logger = logging.getLogger("message_log")
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
        "metadata": metadata or {}
    }
    
    logger.info(json.dumps(log_entry))