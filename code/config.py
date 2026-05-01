"""Configuration and constants for the support triage agent."""

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SUPPORT_TICKETS_DIR = ROOT_DIR / "support_tickets"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"
INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Retrieval Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Safety / Escalation
ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", "0.7"))

# Companies
COMPANIES = ["HackerRank", "Claude", "Visa"]

# Request types
REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

# Status values
STATUSES = ["replied", "escalated"]

# High-risk keywords that trigger escalation
HIGH_RISK_KEYWORDS = [
    "fraud", "identity theft", "stolen", "hack", "security vulnerability",
    "breach", "unauthorized", "illegal", "lawyer", "lawsuit", "police",
    "death", "suicide", "kill", "threat", "terror", "bomb",
    "refund", "chargeback", "dispute", "ban", "terminate",
    "password", "credentials", "login as", "impersonate",
    "delete all", "wipe", "destroy", "drop table", "rm -rf",
    "score", "grade", "unfair", "cheat", "plagiarism",
    "urgent cash", "loan", "debt", "gambling",
]

# Sensitive product areas that often need escalation
SENSITIVE_AREAS = [
    "billing", "payments", "refunds", "account_deletion",
    "security", "fraud", "legal", "compliance", "privacy_breach",
    "access_control", "permissions", "assessment_integrity",
]

# Company inference hints
COMPANY_HINTS = {
    "HackerRank": [
        "hackerrank", "hacker rank", "assessment", "test", "candidate",
        "interview", "coding challenge", "screen", "recruiter", "hiring",
        "mock interview", "certificate", "badge", "submission", "compiler",
        "workspace", "plagiarism", "proctoring", "ats", "integration",
    ],
    "Claude": [
        "claude", "anthropic", "conversation", "chat", "prompt",
        "model", "ai assistant", "api key", "token", "context window",
        "haiku", "sonnet", "opus", "projects", "artifacts",
        "bedrock", "lti", "education", "team", "workspace",
    ],
    "Visa": [
        "visa", "card", "payment", "merchant", "transaction",
        "charge", "debit", "credit", "pin", "atm", "fraud",
        "dispute", "refund", "minimum spend", "contactless",
        "tap to pay", "visa direct", "verified by visa",
    ],
}

# Product area mappings per company (derived from corpus directory structure)
PRODUCT_AREAS = {
    "HackerRank": [
        "screen", "interview", "engage", "skill_up",
        "integrations", "hackerrank_community", "general_help",
        "plagiarism", "proctoring", "ats", "api",
        "billing", "account_management", "workspace",
    ],
    "Claude": [
        "privacy", "billing", "api", "claude_code",
        "claude_desktop", "claude_api_and_console", "amazon_bedrock",
        "claude_for_education", "claude_for_government",
        "account", "security", "projects", "artifacts",
    ],
    "Visa": [
        "payments", "fraud_security", "card_services",
        "merchant_services", "digital_solutions", "support",
    ],
    "None": [
        "general", "billing", "security", "account",
    ],
}

