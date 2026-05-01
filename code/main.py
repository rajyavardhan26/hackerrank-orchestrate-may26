#!/usr/bin/env python3
"""
Support Triage Agent — CLI Entry Point
======================================
Run against support_tickets/support_tickets.csv and write predictions
to support_tickets/output.csv.

Usage:
    python code/main.py
    python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
    python code/main.py --sample  # run on sample data for quick validation
"""

import argparse
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict

# Ensure code/ is on path when running from repo root
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from config import INPUT_CSV, OUTPUT_CSV, SAMPLE_CSV
from agent import TriageAgent
from embedding_store import get_store
from llm_client import get_client

console = Console()


def read_tickets(path: Path) -> List[Dict[str, str]]:
    """Read CSV and return list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_output(path: Path, rows: List[Dict[str, str]]):
    """Write predictions to output CSV."""
    fieldnames = ["issue", "subject", "company", "status", "product_area",
                  "response", "justification", "request_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_agent(input_path: Path, output_path: Path, limit: int = None):
    """Main runner: load data, process tickets, write output."""
    console.print("[bold green]Support Triage Agent[/bold green] — starting up")

    # Verify API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]ERROR:[/bold red] No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment."
        )
        sys.exit(1)

    # Initialize components
    console.print("[cyan]Loading embedding store...[/cyan]")
    store = get_store()

    console.print("[cyan]Initializing LLM client...[/cyan]")
    llm = get_client()

    agent = TriageAgent(llm=llm, store=store)

    # Read tickets
    console.print(f"[cyan]Reading tickets from {input_path}...[/cyan]")
    tickets = read_tickets(input_path)
    if limit:
        tickets = tickets[:limit]
    console.print(f"[cyan]Loaded {len(tickets)} tickets.[/cyan]")

    # Process
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing tickets...", total=len(tickets))
        for ticket in tickets:
            issue = ticket.get("Issue", ticket.get("issue", ""))
            subject = ticket.get("Subject", ticket.get("subject", ""))
            company = ticket.get("Company", ticket.get("company", "None"))

            try:
                prediction = agent.process_ticket(issue, subject, company)
            except Exception as e:
                console.print(f"[yellow]Warning: error on ticket: {e}[/yellow]")
                prediction = {
                    "status": "escalated",
                    "product_area": "general",
                    "response": "We encountered an error processing your request. A human agent will assist you.",
                    "justification": f"System error: {str(e)}",
                    "request_type": "product_issue",
                }

            result = {
                "issue": issue,
                "subject": subject,
                "company": company,
                "status": prediction["status"],
                "product_area": prediction["product_area"],
                "response": prediction["response"],
                "justification": prediction["justification"],
                "request_type": prediction["request_type"],
            }
            results.append(result)
            progress.advance(task)

    # Write output
    write_output(output_path, results)
    console.print(f"[bold green]Done![/bold green] Wrote {len(results)} predictions to {output_path}")

    # Summary table
    summary = Table(title="Prediction Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Count", style="magenta")
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    summary.add_row("Replied", str(replied))
    summary.add_row("Escalated", str(escalated))
    summary.add_row("Total", str(len(results)))
    console.print(summary)

    return results


def main():
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--input", type=Path, default=INPUT_CSV,
                        help="Path to input CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Path to output CSV")
    parser.add_argument("--sample", action="store_true",
                        help="Run on sample_support_tickets.csv instead")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N tickets")
    args = parser.parse_args()

    inp = SAMPLE_CSV if args.sample else args.input
    run_agent(inp, args.output, limit=args.limit)


if __name__ == "__main__":
    main()
