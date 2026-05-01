"""Core triage agent: orchestrates retrieval, reasoning, and response generation."""

import json
from typing import Dict, Any, List, Tuple

from config import TOP_K_RETRIEVAL, ESCALATION_THRESHOLD, PRODUCT_AREAS
from corpus_loader import Chunk
from embedding_store import EmbeddingStore, get_store
from classifier import (
    detect_company,
    classify_request_type,
    assess_risk_level,
    infer_product_area,
)
from llm_client import LLMClient, get_client


class TriageAgent:
    """
    End-to-end support triage agent.
    Pipeline:
      1. Company detection
      2. Request classification + risk assessment
      3. Document retrieval (filtered by company)
      4. LLM reasoning with structured output
      5. Safety / escalation override
    """

    def __init__(self, llm: LLMClient = None, store: EmbeddingStore = None):
        self.llm = llm or get_client()
        self.store = store or get_store()

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert support triage agent for HackerRank, Claude (Anthropic), and Visa.\n"
            "Your job is to analyze a support ticket, retrieve relevant documentation, and produce a structured decision.\n\n"
            "RULES:\n"
            "1. Use ONLY the provided support corpus excerpts to ground your answer.\n"
            "2. If the issue is high-risk (fraud, identity theft, security breach, legal threat, billing dispute, score manipulation, "
            "or anything requiring human judgment), you MUST escalate.\n"
            "3. If the issue is out of scope or unrelated to the three companies, mark as invalid and reply politely.\n"
            "4. Never hallucinate policies, URLs, or procedures not present in the corpus.\n"
            "5. Be concise but helpful. For escalated tickets, explain why briefly.\n"
            "6. The response must be user-facing and professional.\n"
        )

    def _build_user_prompt(self, issue: str, subject: str, company: str,
                           request_type: str, risk_score: float, triggers: list,
                           retrieved: List[Tuple[Chunk, float]]) -> str:
        context_blocks = []
        for chunk, score in retrieved:
            context_blocks.append(
                f"---\nSource: {chunk.company}/{chunk.product_area} (score: {score:.3f})\n{chunk.text}\n---"
            )
        context = "\n".join(context_blocks)

        prompt = (
            f"Ticket Details:\n"
            f"- Company: {company}\n"
            f"- Subject: {subject or '(none)'}\n"
            f"- Issue: {issue}\n"
            f"- Detected Request Type: {request_type}\n"
            f"- Risk Score: {risk_score:.2f}\n"
            f"- Risk Triggers: {', '.join(triggers) if triggers else 'None'}\n\n"
            f"Retrieved Support Corpus Excerpts:\n{context}\n\n"
            f"Based ONLY on the excerpts above, produce a JSON object with these exact keys:\n"
            f'- "status": either "replied" or "escalated"\n'
            f'- "product_area": the most relevant support category\n'
            f'- "response": a helpful, grounded, user-facing response (or escalation message)\n'
            f'- "justification": a concise explanation of why you chose this status and response\n'
            f'- "request_type": one of "product_issue", "feature_request", "bug", "invalid"\n\n'
            f"If risk_score >= {ESCALATION_THRESHOLD}, strongly prefer escalating unless it is a simple FAQ.\n"
            f"If no relevant corpus excerpt applies, escalate rather than guess.\n"
        )
        return prompt

    def process_ticket(self, issue: str, subject: str,
                       explicit_company: str) -> Dict[str, str]:
        """Run the full pipeline on a single ticket."""
        # Step 1: Detect company
        company = detect_company(issue, subject, explicit_company)

        # Step 2: Classify request type
        request_type = classify_request_type(issue, subject)

        # Step 3: Risk assessment
        risk_score, triggers = assess_risk_level(issue, subject)

        # Step 4: Retrieve relevant docs
        query = f"{subject or ''} {issue}"
        retrieved = self.store.search(query, top_k=TOP_K_RETRIEVAL,
                                      company_filter=company if company != "None" else None)

        # If no results for specific company, try all companies
        if not retrieved and company != "None":
            retrieved = self.store.search(query, top_k=TOP_K_RETRIEVAL)

        # Step 5: LLM structured generation
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            issue, subject, company, request_type, risk_score, triggers, retrieved
        )

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["replied", "escalated"]},
                "product_area": {"type": "string"},
                "response": {"type": "string"},
                "justification": {"type": "string"},
                "request_type": {"type": "string", "enum": ["product_issue", "feature_request", "bug", "invalid"]},
            },
            "required": ["status", "product_area", "response", "justification", "request_type"],
        }

        try:
            result = self.llm.structured_chat(system_prompt, user_prompt, schema,
                                               temperature=0.1, max_tokens=1200)
        except Exception as e:
            # Fallback: deterministic escalation on LLM failure
            result = {
                "status": "escalated",
                "product_area": infer_product_area(issue, subject, company, retrieved),
                "response": "We are unable to process this request automatically. A human agent will assist you shortly.",
                "justification": f"LLM generation failed ({str(e)}); escalating as safe fallback.",
                "request_type": request_type,
            }

        # Step 6: Safety override
        result = self._safety_override(
            result, risk_score, triggers, company, retrieved, issue, subject
        )

        # Ensure product_area is sensible
        if not result.get("product_area"):
            result["product_area"] = infer_product_area(issue, subject, company, retrieved)

        return result

    def _safety_override(self, result: Dict[str, Any], risk_score: float,
                         triggers: list, company: str,
                         retrieved: List[Tuple[Chunk, float]],
                         issue: str, subject: str) -> Dict[str, Any]:
        """Override LLM output when safety rules demand escalation."""
        text = f"{subject or ''} {issue or ''}".lower()
        must_escalate = False
        reason = ""

        # Hard escalation rules
        if risk_score >= ESCALATION_THRESHOLD:
            must_escalate = True
            reason = f"Risk score {risk_score:.2f} exceeds threshold; triggers: {', '.join(triggers)}"

        if any(t in text for t in ["identity theft", "stolen identity", "security vulnerability", "bug bounty"]):
            must_escalate = True
            reason = "Sensitive security or fraud issue requires human handling."

        if "increase my score" in text or "change my score" in text or "review my answers" in text:
            must_escalate = True
            reason = "Score manipulation requests must be escalated to preserve assessment integrity."

        if "make visa refund" in text or "ban the seller" in text:
            must_escalate = True
            reason = "Refund and merchant ban requests require human investigation."

        if "reveal" in text and "logic" in text:
            must_escalate = True
            reason = "Attempt to extract internal decision logic; escalate."

        if "delete all files" in text or "rm -rf" in text:
            must_escalate = True
            reason = "Potentially malicious request; escalate immediately."

        if "impersonate" in text or "login as" in text or "access without" in text:
            must_escalate = True
            reason = "Unauthorized access attempt; escalate."

        if "urgent cash" in text or "need cash" in text:
            must_escalate = True
            reason = "Financial distress / cash advance request requires human review."

        if "lawyer" in text or "sue" in text or "legal action" in text:
            must_escalate = True
            reason = "Legal threat requires human escalation."

        # Corpus coverage check: if no relevant docs retrieved and it's not a generic greeting
        if not retrieved and len(text.split()) > 5:
            must_escalate = True
            reason = "No relevant support documentation found; escalating to avoid hallucination."

        if must_escalate:
            result["status"] = "escalated"
            result["justification"] = f"[SAFETY OVERRIDE] {reason}"
            if not result.get("response") or result["status"] == "replied":
                result["response"] = (
                    "Thank you for reaching out. This request requires specialized handling, "
                    "so we have escalated it to a human support agent who will assist you shortly."
                )

        # Ensure request_type consistency
        if result.get("request_type") not in ["product_issue", "feature_request", "bug", "invalid"]:
            result["request_type"] = "product_issue"

        return result


if __name__ == "__main__":
    agent = TriageAgent()
    out = agent.process_ticket(
        "How do I add extra time for a candidate?",
        "Extra time accommodation",
        "HackerRank",
    )
    print(json.dumps(out, indent=2))
