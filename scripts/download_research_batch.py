"""
Batch research downloader — grabs diverse source material while firewall is down.
Targets: arXiv papers, RFC docs, OWASP guides, NIST publications, Python PEPs.

Usage:
    python scripts/download_research_batch.py
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

UA = "JCoder-Research/1.0 (AI code assistant research)"


def _dl(url, dest, label=""):
    """Download a file if not already present."""
    if dest.exists() and dest.stat().st_size > 500:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            if len(data) > 200:
                dest.write_bytes(data)
                print(f"  [OK] {dest.name} ({len(data):,}B) {label}")
                return True
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")
    return False


def download_arxiv_papers(out: Path):
    """Download key AI/ML/RAG papers from arXiv."""
    out.mkdir(parents=True, exist_ok=True)
    papers = [
        ("2005.11401", "RAG_original_Lewis_2020"),
        ("2312.10997", "RAG_survey_2024"),
        ("2404.10981", "RAG_evaluation_2024"),
        ("2310.11511", "self_RAG_2023"),
        ("2401.15884", "RAG_fusion_2024"),
        ("2305.06983", "voyager_LLM_agent_2023"),
        ("2308.12950", "autogen_multi_agent_2023"),
        ("2210.03629", "ReAct_reasoning_acting_2022"),
        ("2305.10601", "tree_of_thoughts_2023"),
        ("2309.17452", "multimodal_RAG_2023"),
        ("2401.08406", "corrective_RAG_2024"),
        ("2312.06648", "dense_retrieval_survey_2023"),
        ("2305.13245", "toolformer_2023"),
        ("2306.06031", "orca_progressive_learning_2023"),
        ("2310.06825", "lora_finetuning_survey_2023"),
        ("2307.09288", "llama2_2023"),
        ("2302.13971", "llama_original_2023"),
        ("2304.08485", "minigpt4_2023"),
        ("2305.18290", "direct_preference_optimization_2023"),
        ("2310.01798", "mistral_7b_2023"),
        ("2403.08295", "gemma_2024"),
        ("2407.21783", "llama3_2024"),
        ("2501.12948", "deepseek_r1_2025"),
        ("2412.15115", "qwen2_5_coder_2024"),
        ("2405.04434", "phi3_2024"),
    ]
    count = 0
    for arxiv_id, name in papers:
        dest = out / f"{name}.pdf"
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if _dl(url, dest, f"arXiv:{arxiv_id}"):
            count += 1
        time.sleep(1.0)  # arXiv rate limit
    print(f"  arXiv: {count}/{len(papers)} papers")
    return count


def download_rfc_docs(out: Path):
    """Download key programming-relevant RFCs."""
    out.mkdir(parents=True, exist_ok=True)
    rfcs = [
        (7230, "HTTP/1.1 Message Syntax"),
        (7231, "HTTP/1.1 Semantics"),
        (7540, "HTTP/2"),
        (9110, "HTTP Semantics"),
        (9114, "HTTP/3"),
        (8259, "JSON"),
        (6749, "OAuth 2.0"),
        (7519, "JWT"),
        (5246, "TLS 1.2"),
        (8446, "TLS 1.3"),
        (793,  "TCP"),
        (768,  "UDP"),
        (2616, "HTTP/1.1 Original"),
        (3986, "URI"),
        (6455, "WebSocket"),
        (7807, "Problem Details for HTTP APIs"),
        (9457, "Problem Details Updated"),
        (6570, "URI Template"),
        (2818, "HTTP Over TLS"),
        (4648, "URI Internationalization"),
    ]
    count = 0
    for num, title in rfcs:
        dest = out / f"rfc{num}.txt"
        url = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"
        if _dl(url, dest, title):
            count += 1
        time.sleep(0.3)
    print(f"  RFCs: {count}/{len(rfcs)}")
    return count


def download_python_peps(out: Path):
    """Download key Python Enhancement Proposals."""
    out.mkdir(parents=True, exist_ok=True)
    peps = [8, 20, 257, 484, 526, 557, 572, 604, 612, 616, 617, 618,
            634, 636, 646, 654, 657, 673, 681, 695, 696, 698, 702, 703, 709, 718, 742, 750]
    count = 0
    for num in peps:
        dest = out / f"pep-{num:04d}.rst"
        url = f"https://peps.python.org/pep-{num:04d}/"
        if _dl(url, dest, f"PEP {num}"):
            count += 1
        time.sleep(0.3)
    print(f"  PEPs: {count}/{len(peps)}")
    return count


def download_nist_pubs(out: Path):
    """Download key NIST cybersecurity publications."""
    out.mkdir(parents=True, exist_ok=True)
    pubs = [
        ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf", "NIST_SP_800-53r5_security_controls.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-171r3.pdf", "NIST_SP_800-171r3_CUI_protection.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-4.pdf", "NIST_SP_800-63-4_digital_identity.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "NIST_AI_RMF_100-1.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf", "NIST_AI_GenAI_600-1.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1300.pdf", "NIST_CSF_2.0_SP1300.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf", "NIST_SP_800-61r3_incident_response.pdf"),
    ]
    count = 0
    for url, name in pubs:
        dest = out / name
        if _dl(url, dest, "NIST"):
            count += 1
        time.sleep(0.5)
    print(f"  NIST: {count}/{len(pubs)}")
    return count


def download_crs_reports(out: Path):
    """Download Congressional Research Service reports index + samples."""
    out.mkdir(parents=True, exist_ok=True)
    # CRS reports CSV index
    _dl("https://www.everycrsreport.com/reports.csv", out / "crs_reports_index.csv", "CRS index")
    # Sample recent reports
    samples = [
        "https://crsreports.congress.gov/product/pdf/R/R47644",
        "https://crsreports.congress.gov/product/pdf/R/R47924",
        "https://crsreports.congress.gov/product/pdf/IF/IF12525",
        "https://crsreports.congress.gov/product/pdf/R/R47569",
        "https://crsreports.congress.gov/product/pdf/R/R47708",
    ]
    count = 0
    for i, url in enumerate(samples):
        dest = out / f"crs_report_{i+1}.pdf"
        if _dl(url, dest, "CRS"):
            count += 1
        time.sleep(0.5)
    print(f"  CRS: {count}/{len(samples)} + index CSV")
    return count


def main():
    base = Path("data/raw_downloads/research")
    base.mkdir(parents=True, exist_ok=True)
    total = 0

    print("=== Batch Research Download ===\n")

    print("[1/5] arXiv AI/ML papers...")
    total += download_arxiv_papers(base / "arxiv_papers")

    print("\n[2/5] IETF RFCs...")
    total += download_rfc_docs(base / "rfc_docs")

    print("\n[3/5] Python PEPs...")
    total += download_python_peps(base / "python_peps")

    print("\n[4/5] NIST publications...")
    total += download_nist_pubs(base / "nist_pubs")

    print("\n[5/5] CRS reports...")
    total += download_crs_reports(base / "crs_reports")

    print(f"\n=== DONE: {total} files downloaded to {base} ===")


if __name__ == "__main__":
    os.chdir(os.environ.get("JCODER_ROOT", str(Path(__file__).resolve().parent.parent)))
    main()
