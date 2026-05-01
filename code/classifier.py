"""Company detection and request classification."""

import re
from typing import Tuple, Optional

from config import COMPANIES, REQUEST_TYPES, COMPANY_HINTS, HIGH_RISK_KEYWORDS


def detect_company(issue: str, subject: str, explicit_company: str) -> str:
    """
    Determine which company the ticket belongs to.
    Returns one of: HackerRank, Claude, Visa, None
    """
    if explicit_company and explicit_company in COMPANIES:
        return explicit_company

    text = f"{subject or ''} {issue or ''}".lower()
    scores = {}
    for company, hints in COMPANY_HINTS.items():
        score = sum(1 for hint in hints if hint.lower() in text)
        scores[company] = score

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "None"


def classify_request_type(issue: str, subject: str) -> str:
    """
    Classify the request into one of: product_issue, feature_request, bug, invalid
    """
    text = f"{subject or ''} {issue or ''}".lower()

    # Invalid detection
    invalid_signals = [
        "iron man", "actor", "movie", "weather", "news",
        "who is", "what is the capital", "sports", "politics",
        "not working" in text and len(text.split()) < 5,
        "help" in text and len(text.split()) < 4,
    ]
    if any(s for s in invalid_signals if isinstance(s, bool) and s):
        return "invalid"
    if any(s.lower() in text for s in invalid_signals if isinstance(s, str)):
        return "invalid"

    # Bug detection
    bug_patterns = [
        r"\b(bug|glitch|broken|crash|error|not working|stopped|down|fail|failing)\b",
        r"\b(site is down|page not loading|can't access|unable to)\b",
        r"\b(compatibility|compatible check|system error)\b",
    ]
    for pattern in bug_patterns:
        if re.search(pattern, text):
            return "bug"

    # Feature request detection
    feature_signals = [
        "can you add", "would be great", "feature request", "please add",
        "i want", "i would like", "suggestion", "improvement",
        "ability to", "option to", "support for",
    ]
    if any(sig in text for sig in feature_signals):
        return "feature_request"

    # Default to product_issue for most support queries
    return "product_issue"


def assess_risk_level(issue: str, subject: str) -> Tuple[float, list]:
    """
    Assess risk score (0.0 - 1.0) and return triggering keywords.
    Higher = more likely to need escalation.
    """
    text = f"{subject or ''} {issue or ''}".lower()
    triggers = []
    score = 0.0

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword.lower() in text:
            triggers.append(keyword)
            score += 0.15

    # Additional heuristics
    if re.search(r"\b(urgent|asap|immediately|today|now)\b", text):
        score += 0.1
    if re.search(r"\b(lawyer|legal|sue|court|police)\b", text):
        score += 0.25
    if re.search(r"\b(refund.*ban|ban.*refund|make.*refund)\b", text):
        score += 0.2
    if re.search(r"\b(increase.*score|change.*score|review.*answer)\b", text):
        score += 0.2
    if re.search(r"\b(delete.*account|remove.*data|wipe)\b", text):
        score += 0.1
    if re.search(r"\b(security vulnerability|bug bounty|exploit|hack)\b", text):
        score += 0.25
    if re.search(r"\b(impersonate|login as|access.*without|bypass)\b", text):
        score += 0.3
    if re.search(r"\b(cash advance|urgent cash|loan|gambling)\b", text):
        score += 0.2
    if re.search(r"\b(reveal.*logic|show.*rules|internal.*document)\b", text):
        score += 0.25
    if re.search(r"\b(delete all files|rm -rf|drop table|destroy)\b", text):
        score += 0.3

    # Penalize very short / vague tickets
    word_count = len(text.split())
    if word_count < 5:
        score += 0.1

    return min(score, 1.0), triggers


def infer_product_area(issue: str, subject: str, company: str,
                       retrieved_chunks: list) -> str:
    """Infer the most relevant product area from retrieved chunks and heuristics."""
    text = f"{subject or ''} {issue or ''}".lower()

    # Use retrieved chunk product areas as signal
    area_scores = {}
    for chunk, score in retrieved_chunks:
        area = chunk.product_area
        area_scores[area] = area_scores.get(area, 0) + score

    # Boost based on keyword matching
    area_keywords = {
        "screen": ["test", "assessment", "candidate", "invite", "question"],
        "interview": ["interview", "live", "pair", "coding interview", "panel"],
        "engage": ["survey", "feedback", "engagement", "pulse"],
        "billing": ["subscription", "invoice", "payment", "charge", "refund", "plan"],
        "account_management": ["user", "admin", "role", "permission", "seat", "team"],
        "privacy": ["delete", "data", "conversation", "privacy", "gdpr"],
        "security": ["fraud", "theft", "vulnerability", "breach", "hack"],
        "api": ["api", "integration", "webhook", "developer"],
        "claude_code": ["claude code", "ide", "vscode", "cursor"],
        "claude_desktop": ["desktop app", "mac", "windows app"],
        "hackerrank_community": ["community", "profile", "badge", "certificate"],
        "general_help": ["help", "how to", "guide", "tutorial"],
    }

    for area, keywords in area_keywords.items():
        for kw in keywords:
            if kw in text:
                area_scores[area] = area_scores.get(area, 0) + 1.0

    if area_scores:
        best_area = max(area_scores, key=area_scores.get)
        return best_area

    # Fallback
    if company == "HackerRank":
        return "general_help"
    elif company == "Claude":
        return "privacy"
    elif company == "Visa":
        return "support"
    return "general"


if __name__ == "__main__":
    issue = "I can't access my test. The site says error 500."
    subj = "Test access issue"
    print(detect_company(issue, subj, "None"))
    print(classify_request_type(issue, subj))
    print(assess_risk_level(issue, subj))
