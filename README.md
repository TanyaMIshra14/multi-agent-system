Overview
This project is a Python-based multi-agent framework designed to classify, analyze, and extract information from various document types such as JSON, Emails, and PDF-like text. It includes a shared memory layer using SQLite to persist context and agent logs across operations.

Features
 Classifier Agent: Detects the document type (PDF, JSON, Email) and classifies its intent (Invoice, RFQ, Complaint, Regulation, etc.).
PDF Agent: Extracts fields like invoice numbers, total amounts, dates, and regulatory sections from PDF-like text.
Email Agent: Extracts metadata, urgency level, and intent-specific information from raw email text.
JSON Agent: Validates schema and extracts structured fields based on intent from JSON input.
Shared Memory: Maintains processing context and logs using a lightweight SQLite database.
Multi-Agent Orchestration: Central engine routes the input to the appropriate agent based on the classifier's output.

Tech Stack
Python 3.10+
SQLite (via sqlite3)
asyncio for asynchronous processing
dataclasses, enum, uuid, and logging

How It Works
Input (JSON, Email string, or PDF-like text) is passed to the MultiAgentOrchestrator
The ClassifierAgent detects the document type and intent.
Based on the result, the corresponding agent (JSONAgent, EmailAgent, or PDFAgent) processes the data.
Context and actions are saved in SQLite for traceability and history lookup.
