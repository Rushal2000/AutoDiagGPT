
from dotenv import load_dotenv
import os
os.environ["OLLAMA_NUM_GPU"] = "0"

# Load environment variables
load_dotenv()

#  API Keys 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

#  Provider 
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

#  Models 
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

#  Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "extracted_images")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
OCR_FILE = os.path.join(BASE_DIR, "ocr_results.json")
CAPTION_FILE = os.path.join(BASE_DIR, "diagram_captions.json")

# RAG Settings 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

#  System Keywords for Auto-Tagging 
SYSTEM_KEYWORDS = {
    "hydraulics": [
        "hydraulic", "hydraulic oil", "hydraulic tank",
        "hydraulic circuit", "hydraulic pump", "pressure",
        "flow", "hose", "cylinder", "boom", "arm",
        "bucket", "work equipment", "relief valve",
        "oil leakage", "hydraulic breaker"
    ],

    "engine": [
        "engine", "starting engine", "engine oil",
        "coolant", "radiator", "air cleaner",
        "fuel system", "idle", "engine speed",
        "overheating", "exhaust", "fuel filter"
    ],

    "electrical": [
        "electric system", "electrical system",
        "battery", "fuse", "fusible link",
        "wiring", "connector", "switch",
        "starter", "alternator", "voltage",
        "warning buzzer", "monitor panel"
    ],

    "monitoring": [
        "monitor", "machine monitor",
        "warning lamp", "alarm", "gauge",
        "indicator", "self diagnosis",
        "display", "warning buzzer"
    ],

    "controls": [
        "control lever", "joystick",
        "pedal", "switch", "lock lever",
        "safety lock lever", "travel lever",
        "working mode", "active mode"
    ],

    "travel_system": [
        "travel", "track", "track shoe",
        "sprocket", "undercarriage",
        "travel motor", "travel speed",
        "moving machine", "swing",
        "loading", "transportation"
    ],

    "steering": [
        "steering", "steering system",
        "steering machine", "travel lever",
        "direction change", "turning",
        "travel control"
    ],

    "brakes": [
        "brake", "parking brake",
        "service brake", "braking",
        "brake system"
    ],

    "battery": [
        "battery", "electrolyte",
        "charging battery", "booster cable",
        "battery terminal", "battery acid",
        "hydrogen gas"
    ],

    "safety": [
        "safety", "warning", "danger",
        "caution", "lockout", "interlock",
        "emergency exit", "fire extinguisher",
        "seat belt", "fops", "rops",
        "high voltage", "safety lock lever",
        "do not operate"
    ],

    "attachments": [
        "attachment", "bucket",
        "bucket with hook", "breaker",
        "crusher", "cutter", "grapple",
        "fork grab", "power ripper",
        "hydraulic pile driver"
    ],

    "maintenance": [
        "maintenance", "inspection",
        "service", "lubrication",
        "replacement", "wear parts",
        "troubleshooting", "repair",
        "periodic service", "grease",
        "oil change"
    ]
}


def detect_system(text: str) -> str:
    """Auto-detect which vehicle system a text chunk belongs to."""
    text_lower = text.lower()
    scores = {}
    for system, keywords in SYSTEM_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[system] = score
    return max(scores, key=scores.get) if scores else "general"


#  Create directories if they don't exist 
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
