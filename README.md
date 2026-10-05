# RAG Trust Scorer

This repository contains the code for our COMPSCI 764 Final Project. It implements a lightweight "Trust Scorer" to defend Retrieval-Augmented Generation (RAG) systems against indirect prompt injection attacks.

## Project Summary
Instead of relying on large language models to filter malicious contexts, our system extracts 8 features (such as Perplexity and Directive Density) and uses a **Dual-Gated Abstention Policy** to filter out poisoned data before it reaches the LLM. 

**Key Results:**
- Mitigated 88.75% of overall attacks.
- Intercepted 100% of seen attacks (0.00% residual ASR).
- Blocked 85% of unseen zero-shot attacks (15.00% residual ASR).
- Retained 97.78% of benign utility.

## How to Run

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
