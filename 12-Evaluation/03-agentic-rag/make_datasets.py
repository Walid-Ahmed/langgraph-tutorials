# Build the chatbot's company files: 6 CSV exports and 5 proposal PDFs under datasets/.
#
# The company and every record in it are fictional, written for this book. The
# values are chosen so the 18 golden cases in 04-agentic-rag-evaluation have
# true answers (Sarah Lee covers Europe; the completed projects' budgets sum to
# $2,150,000; and so on), plus enough distractors - inactive accounts, open and
# lost deals, on-budget projects - that the agent has to find the right rows.
#
# The PDFs are written with a minimal built-in writer and a standard font, so
# they need no extra package and pypdf reads their text back exactly.
#
# Run it with (from 12-Evaluation/03-agentic-rag):
#    python make_datasets.py
# Expected result: "wrote 11 files to datasets/". Delete .chroma/ afterwards so
# the next start re-ingests the new files.

import csv
from pathlib import Path

DATASETS = Path(__file__).resolve().parent / "datasets"

SALES_REPS = [
    ("REP01", "Sarah Lee", "Europe", "sarah.lee@example.com"),
    ("REP02", "Michael Grant", "North America", "michael.grant@example.com"),
    ("REP03", "Aisha Rahman", "APAC", "aisha.rahman@example.com"),
    ("REP04", "Tom Becker", "Middle East & Africa", "tom.becker@example.com"),
]

ACCOUNTS = [  # account_id, account_name, industry, region, status, sales_rep_id
    ("ACC001", "HealthFirst", "Healthcare", "North America", "Active", "REP02"),
    ("ACC002", "RetailMax", "Retail", "North America", "Active", "REP02"),
    ("ACC003", "MedCore", "Healthcare", "Europe", "Active", "REP01"),
    ("ACC004", "FinTrust", "Finance", "North America", "Active", "REP02"),
    ("ACC005", "TechNova", "Technology", "North America", "Active", "REP02"),
    ("ACC006", "AutoWorks", "Manufacturing", "Europe", "Active", "REP01"),
    ("ACC007", "Lumen Bank", "Finance", "Europe", "Active", "REP01"),
    ("ACC008", "Orbit Telecom", "Telecommunications", "APAC", "Active", "REP03"),
    ("ACC009", "Northwind Logistics", "Logistics", "North America", "Inactive", "REP02"),
    ("ACC010", "Sahara Foods", "Retail", "Middle East & Africa", "Active", "REP04"),
]

OPPORTUNITIES = [  # opportunity_id, account_id, account_name, opportunity_name, stage, status, deal_value, sales_rep_id
    ("OPP001", "ACC001", "HealthFirst", "AWS Migration", "Negotiation", "Lost", 500000, "REP02"),
    ("OPP002", "ACC004", "FinTrust", "Azure Modernization", "Closed", "Won", 820000, "REP02"),
    ("OPP003", "ACC007", "Lumen Bank", "Snowflake Platform", "Closed", "Won", 660000, "REP01"),
    ("OPP004", "ACC008", "Orbit Telecom", "Databricks Analytics", "Closed", "Won", 710000, "REP03"),
    ("OPP005", "ACC005", "TechNova", "GraphRAG Platform", "Closed", "Won", 450000, "REP02"),
    ("OPP006", "ACC006", "AutoWorks", "Factory AI Insights", "Closed", "Won", 390000, "REP01"),
    ("OPP007", "ACC002", "RetailMax", "Retail Recommendation Engine", "Closed", "Won", 420000, "REP02"),
    ("OPP008", "ACC003", "MedCore", "Clinical Data Lake", "Proposal", "Open", 600000, "REP01"),
    ("OPP009", "ACC001", "HealthFirst", "Patient Portal Upgrade", "Qualification", "Open", 250000, "REP02"),
    ("OPP010", "ACC008", "Orbit Telecom", "Churn Prediction Model", "Proposal", "Lost", 180000, "REP03"),
    ("OPP011", "ACC010", "Sahara Foods", "Supply Chain Dashboard", "Discovery", "Open", 150000, "REP04"),
]

PROJECTS = [  # project_id, project_name, account_id, technology, status, budget, actual_cost, delay_days
    ("PRJ001", "Azure Modernization", "ACC004", "Azure", "Completed", 800000, 760000, 0),
    ("PRJ002", "Snowflake Platform", "ACC007", "Snowflake", "Completed", 650000, 640000, 0),
    ("PRJ003", "Databricks Analytics", "ACC008", "Databricks", "Completed", 700000, 690000, 0),
    ("PRJ004", "GraphRAG Platform", "ACC005", "Neo4j, LangGraph, OpenAI", "Delayed", 450000, 520000, 35),
    ("PRJ005", "Factory AI Insights", "ACC006", "AWS IoT, PyTorch", "Delayed", 380000, 430000, 45),
    ("PRJ006", "Retail Recommendation Engine", "ACC002", "Google Cloud Vertex AI", "In Progress", 400000, 310000, 0),
]

PROJECT_RESOURCES = [  # project_id, employee_name, role, allocation_percent
    ("PRJ001", "Emma Davis", "Project Manager", 50),
    ("PRJ001", "Omar Haddad", "Cloud Architect", 100),
    ("PRJ002", "Lucas Moreau", "Data Engineer", 100),
    ("PRJ003", "Mei Tanaka", "Analytics Lead", 80),
    ("PRJ004", "Alex Martin", "Lead Engineer", 100),
    ("PRJ005", "Jonas Weber", "ML Engineer", 100),
    ("PRJ005", "Fatima Zahra", "IoT Specialist", 60),
    ("PRJ006", "Carlos Ruiz", "ML Engineer", 100),
]

REVENUE = [  # quarter, industry, revenue
    ("2025-Q1", "Healthcare", 1800000), ("2025-Q1", "Finance", 1250000), ("2025-Q1", "Technology", 1100000),
    ("2025-Q1", "Retail", 950000), ("2025-Q1", "Manufacturing", 720000),
    ("2025-Q2", "Healthcare", 900000), ("2025-Q2", "Finance", 1400000), ("2025-Q2", "Technology", 1300000),
    ("2025-Q2", "Retail", 1050000), ("2025-Q2", "Manufacturing", 800000),
    ("2025-Q3", "Healthcare", 1150000), ("2025-Q3", "Finance", 1350000), ("2025-Q3", "Technology", 1500000),
    ("2025-Q3", "Retail", 990000), ("2025-Q3", "Manufacturing", 860000),
]

PROPOSALS = {  # file name -> lines of text
    "proposal_001.pdf": [
        "Proposal 001 - HealthFirst: AWS Migration", "Client: HealthFirst (ACC001), Healthcare, North America",
        "Date: 2025-02-10", "Proposal value: $500,000",
        "Scope: move HealthFirst's patient-scheduling and billing systems to AWS.",
        "Timeline: 9 months.", "Main risk: data residency rules for patient records.",
        "Status: Lost at the Negotiation stage.",
    ],
    "proposal_002.pdf": [
        "Proposal 002 - MedCore: Clinical Data Lake", "Client: MedCore (ACC003), Healthcare, Europe",
        "Date: 2025-06-02", "Proposal value: $600,000",
        "Scope: a governed data lake for MedCore's clinical-trial data.",
        "Timeline: 12 months.", "Main risk: proposal review delayed by 3 weeks, pushing the start date.",
        "Status: Under review.",
    ],
    "proposal_003.pdf": [
        "Proposal 003 - FinTrust: Azure Modernization", "Client: FinTrust (ACC004), Finance, North America",
        "Date: 2025-01-20", "Proposal value: $820,000",
        "Scope: modernize FinTrust's core banking services on Azure.",
        "Timeline: 10 months.", "Main risk: legacy integration with the mainframe ledger.",
        "Status: Won.",
    ],
    "proposal_004.pdf": [
        "Proposal 004 - TechNova: GraphRAG Platform", "Client: TechNova (ACC005), Technology, North America",
        "Date: 2025-03-14", "Proposal value: $450,000",
        "Scope: a question-answering platform over TechNova's product knowledge graph.",
        "Technologies: Neo4j, LangGraph and OpenAI.", "Timeline: 6 months.",
        "Main risk: graph data quality.", "Status: Won.",
    ],
    "proposal_005.pdf": [
        "Proposal 005 - AutoWorks: Factory AI Insights", "Client: AutoWorks (ACC006), Manufacturing, Europe",
        "Date: 2025-04-08", "Proposal value: $390,000",
        "Service: Factory AI Insights, predictive maintenance and quality analytics",
        "for AutoWorks' assembly lines, built on sensor data.",
        "Timeline: 8 months.", "Main risk: sensor coverage on older lines.", "Status: Won.",
    ],
}


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_pdf(path: Path, lines: list[str]) -> None:
    """A one-page PDF with plain text in Helvetica, one of the 14 standard PDF fonts."""
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_ops = ["BT", "/F1 16 Tf", "72 760 Td", f"({escape(lines[0])}) Tj", "/F1 11 Tf"]
    text_ops += [f"0 -22 Td ({escape(line)}) Tj" for line in lines[1:]]
    stream = "\n".join(text_ops + ["ET"]).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def main() -> None:
    write_csv(DATASETS / "crm_exports" / "sales_reps.csv", ["sales_rep_id", "name", "region", "email"], SALES_REPS)
    write_csv(DATASETS / "crm_exports" / "accounts.csv",
              ["account_id", "account_name", "industry", "region", "status", "sales_rep_id"], ACCOUNTS)
    write_csv(DATASETS / "crm_exports" / "opportunities.csv",
              ["opportunity_id", "account_id", "account_name", "opportunity_name", "stage", "status", "deal_value",
               "sales_rep_id"],
              OPPORTUNITIES)
    write_csv(DATASETS / "projects" / "projects.csv",
              ["project_id", "project_name", "account_id", "technology", "status", "budget", "actual_cost",
               "delay_days"], PROJECTS)
    write_csv(DATASETS / "projects" / "project_resources.csv",
              ["project_id", "employee_name", "role", "allocation_percent"], PROJECT_RESOURCES)
    write_csv(DATASETS / "revenue" / "revenue.csv", ["quarter", "industry", "revenue"], REVENUE)
    for name, lines in PROPOSALS.items():
        write_pdf(DATASETS / "proposal_documents" / name, lines)
    print(f"wrote {6 + len(PROPOSALS)} files to datasets/")


if __name__ == "__main__":
    main()
