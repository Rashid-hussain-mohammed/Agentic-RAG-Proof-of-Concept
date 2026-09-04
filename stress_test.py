import os
import sys
import time
from tabulate import tabulate
from fpdf import FPDF
from app.rag.ingestor import process_and_index_document
from app.services.agent import app_agent

TEST_SUITE = [
    # Factual Extraction
    {
        "category": "Factual",
        "question": "What is Apple's state of incorporation and where is its principal executive office located?",
        "expected": "Direct answer (1 loop)."
    },
    {
        "category": "Factual",
        "question": "What was Apple's total net sales for fiscal year 2023?",
        "expected": "Direct answer (1 loop)."
    },
    {
        "category": "Factual",
        "question": "What fiscal year-end date does this filing cover?",
        "expected": "Direct answer (1 loop)."
    },
    
    # Numerical / Table Extraction
    {
        "category": "Table Extraction",
        "question": "What were iPhone net sales in fiscal 2023, and how did that compare to fiscal 2022?",
        "expected": "Requires reading across table columns."
    },
    {
        "category": "Table Extraction",
        "question": "What was Mac's year-over-year percentage change in net sales for 2023?",
        "expected": "Requires extracting and calculating or finding the explicit % column."
    },
    {
        "category": "Table Extraction",
        "question": "What was the Services segment's gross margin percentage in fiscal 2023?",
        "expected": "Extracting specific margin data from the Services row."
    },
    {
        "category": "Table Extraction",
        "question": "What were net sales by region (Americas, Europe, Greater China, Japan, Rest of Asia Pacific) for fiscal 2023?",
        "expected": "Multi-entity extraction from the regional breakdown table."
    },

    # Multi-hop Reasoning
    {
        "category": "Multi-Hop",
        "question": "Which product categories declined in fiscal 2023, and what does the filing say caused Mac's decline specifically?",
        "expected": "Requires identifying negative growth categories, then finding the MD&A narrative for Mac."
    },
    {
        "category": "Multi-Hop",
        "question": "How does Services gross margin compare to Products gross margin, and what narrative explanation does the MD&A give for the gap?",
        "expected": "Compare two numbers, then retrieve qualitative context."
    },
    {
        "category": "Multi-Hop",
        "question": "The filing attributes the total net sales decrease to certain factors — which single item accounted for more than the entire year-over-year decrease?",
        "expected": "Requires analyzing the narrative breakdown of revenue declines."
    },

    # Cross-referencing
    {
        "category": "Cross-Reference",
        "question": "Find the App Store/antitrust litigation discussion — is it referenced in both the Legal Proceedings section and elsewhere (e.g., risk factors)? What's added in each place?",
        "expected": "Tests retrieval spread and chunk merging."
    },
    {
        "category": "Cross-Reference",
        "question": "Is the iPhone net sales figure reported consistently between the 'Net sales by category' table and the narrative discussion right after it?",
        "expected": "Tests data consistency across different chunks."
    },

    # Negation / Exclusion
    {
        "category": "Negation",
        "question": "Which region's net sales did NOT decrease in fiscal 2023?",
        "expected": "Requires evaluating all regions and finding the exception."
    },
    {
        "category": "Negation",
        "question": "Which product category's decline was NOT attributed primarily to currency effects?",
        "expected": "Filters out 'currency effects' from MD&A explanations."
    },

    # Out-of-scope / Hallucination Traps
    {
        "category": "Out-of-Scope",
        "question": "What was AAPL's closing stock price on the last trading day of fiscal 2023?",
        "expected": "REFUSAL: Not explicitly in a standard 10-K text."
    },
    {
        "category": "Out-of-Scope",
        "question": "How many Apple Stores are there in Germany specifically?",
        "expected": "REFUSAL: Not broken out at that granularity."
    },
    {
        "category": "Out-of-Scope",
        "question": "What is Apple's revenue guidance for fiscal 2024?",
        "expected": "REFUSAL: 10-Ks do not provide forward guidance."
    },

    # Ambiguous / Underspecified
    {
        "category": "Ambiguous",
        "question": "What was the revenue?",
        "expected": "Agent should attempt to find Total Net Sales or ask for clarification."
    },
    {
        "category": "Ambiguous",
        "question": "What are the risks?",
        "expected": "Agent should summarize the Risk Factors section broadly."
    },

    # Paraphrase Robustness
    {
        "category": "Paraphrase",
        "question": "How much did Apple bring in across all product and service lines combined in FY2023?",
        "expected": "Tests semantic match to 'Total net sales'."
    },
    {
        "category": "Paraphrase",
        "question": "What share of Apple's total revenue came from smartphone sales?",
        "expected": "Requires mapping 'smartphone' to 'iPhone' and dividing by total sales."
    },

    # Summarization
    {
        "category": "Summarization",
        "question": "Summarize the reasons given for the year-over-year decline in Mac net sales.",
        "expected": "Should pull multi-sentence qualitative reasons into a concise list."
    },
    {
        "category": "Summarization",
        "question": "Summarize the regional performance section — which regions grew and which declined, and why?",
        "expected": "Requires heavy chunk aggregation."
    },

    # Definitional / Footnote Precision
    {
        "category": "Footnotes",
        "question": "According to the filing's footnotes, what does 'Services net sales' include beyond direct service purchases?",
        "expected": "Tests ability to retrieve small footnote text (e.g., AppleCare, advertising)."
    },
    {
        "category": "Footnotes",
        "question": "What specific iPhone models are described as comprising the iPhone product line in the terminal?",
        "expected": "Tests granular product description retrieval."
    }
]

def sanitize_text(text: str) -> str:
    """Replaces Unicode characters with ASCII equivalents for fpdf compatibility."""
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2022": "*", "\u2026": "...",
        "✅": "[PASS]", "❌": "[FAIL]", "⚠️": "[WARN]", "💥": "[ERR]"
    }
    for orig, target in replacements.items():
        text = text.replace(orig, target)
    return text.encode("latin-1", "replace").decode("latin-1")

def save_to_pdf(results, filename="stress_test_report.pdf"):
    pdf = FPDF(orientation="L")
    pdf.add_page()
    
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "Agentic RAG Stress Test Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("helvetica", size=8)
    headers = ["Category", "Time", "Loops", "Verdict", "Agent Output Snippet"]
    
    with pdf.table(text_align="LEFT") as table:
        header_row = table.row()
        for header in headers:
            header_row.cell(header)
            
        for result in results:
            row = table.row()
            for item in result:
                row.cell(sanitize_text(str(item)))
                
    pdf.output(filename)
    print(f"\n📄 PDF report successfully saved to {filename}")

def run_stress_test(pdf_path: str = None):
    print("=" * 80)
    print("🚀 STARTING AGENTIC RAG STRESS TEST (25-QUESTION BATTERY)")
    print("=" * 80)

    if pdf_path and os.path.exists(pdf_path):
        print(f"\n[1/2] Ingesting document: {pdf_path}")
        chunk_count = process_and_index_document(pdf_path)
        print(f"      Successfully indexed {chunk_count} chunks into ChromaDB.")
    else:
        print("\n[1/2] No new PDF specified or file not found. Testing existing vector store.")

    print("\n[2/2] Executing queries...\n")
    results = []

    for idx, test in enumerate(TEST_SUITE, start=1):
        print(f"Running Test {idx}/{len(TEST_SUITE)}: [{test['category']}]")
        print(f"Q: {test['question']}")
        
        start_time = time.time()
        initial_state = {
            "question": test["question"],
            "context": "",
            "answer": "",
            "generation_ready": "",
            "loop_count": 0
        }
        
        try:
            output = app_agent.invoke(initial_state)
            elapsed = time.time() - start_time
            
            answer = output.get("answer", "").strip()
            loops = output.get("loop_count", 1)
            
            is_refusal = any(phrase in answer.lower() for phrase in [
                "do not know", "does not contain", "not mentioned", "not provided", "cannot answer"
            ])
            
            if test["category"] == "Out-of-Scope":
                status = "✅ PASS (Refused)" if is_refusal else "❌ FAIL (Hallucinated)"
            else:
                status = "✅ PASS (Answered)" if not is_refusal else "⚠️ WARN (Refused)"

            clean_snippet = answer.replace("\n", " ")[:65] + ("..." if len(answer) > 65 else "")
            
            results.append([
                f"T{idx}: {test['category']}",
                f"{elapsed:.2f}s",
                loops,
                status,
                clean_snippet
            ])
            print(f"Result: {status} in {elapsed:.2f}s ({loops} loops)\n")

        except Exception as e:
            results.append([f"T{idx}: {test['category']}", "ERR", "ERR", "💥 CRASH", str(e)[:60]])
            print(f"Test crashed: {str(e)}\n")

    headers = ["Category", "Time", "Loops", "Verdict", "Agent Output Snippet"]
    print("=" * 80)
    print("📊 STRESS TEST REPORT")
    print("=" * 80)
    print(tabulate(results, headers=headers, tablefmt="grid", maxcolwidths=[20, 8, 5, 20, 70]))

    save_to_pdf(results)

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_stress_test(target_file)