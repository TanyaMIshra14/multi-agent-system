import json
import sqlite3
import asyncio
import logging
import uuid
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    PDF = "PDF"
    JSON = "JSON"
    EMAIL = "EMAIL"
    UNKNOWN = "UNKNOWN"


class Intent(Enum):
    INVOICE = "INVOICE"
    RFQ = "RFQ"
    COMPLAINT = "COMPLAINT"
    REGULATION = "REGULATION"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ContextProcessing:
    id: str
    source_type: DocumentType
    intent: Intent
    timestamp: datetime
    sender: Optional[str] = None
    topic: Optional[str] = None
    extracted_fields: Dict[str, Any] = None
    history_processing: List[str] = None

    def __post_init__(self):
        if self.extracted_fields is None:
            self.extracted_fields = {}
        if self.history_processing is None:
            self.history_processing = []


class SharedMemory:
    def __init__(self, db_path: str = "ag_memory.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_contexts(
                id TEXT PRIMARY KEY,
                source_type TEXT,
                intent TEXT,
                timestamp TEXT,
                sender TEXT,
                topic TEXT,
                extracted_fields TEXT,
                history_processing TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_logs(
                id TEXT PRIMARY KEY,
                context_id TEXT,
                agent_name TEXT,
                action TEXT,
                timestamp TEXT,
                details TEXT,
                FOREIGN KEY (context_id) REFERENCES processing_contexts (id)
            )
        ''')

        connection.commit()
        connection.close()

    def store_context(self, context: ContextProcessing):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO processing_contexts
            (id, source_type, intent, timestamp, sender, topic, extracted_fields, history_processing)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
            context.id,
            context.source_type.value,
            context.intent.value,
            context.timestamp.isoformat(),
            context.sender,
            context.topic,
            json.dumps(context.extracted_fields),
            json.dumps(context.history_processing)
        ))

        connection.commit()
        connection.close()

    def fetch_context(self, context_id: str) -> Optional[ContextProcessing]:
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM processing_contexts WHERE id = ?', (context_id,))
        row = cursor.fetchone()
        connection.close()

        if row:
            return ContextProcessing(
                id=row[0],
                source_type=DocumentType(row[1]),
                intent=Intent(row[2]),
                timestamp=datetime.fromisoformat(row[3]),
                sender=row[4],
                topic=row[5],
                extracted_fields=json.loads(row[6]) if row[6] else {},
                history_processing=json.loads(row[7]) if row[7] else []
            )
        return None

    def log_agent_action(self, context_id: str, agent_name: str, action: str, details: Dict[str, Any]):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        log_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO agent_logs(id, context_id, agent_name, action, timestamp, details)
            VALUES(?, ?, ?, ?, ?, ?)
            ''', (
            log_id,
            context_id,
            agent_name,
            action,
            datetime.now().isoformat(),
            json.dumps(details)
        ))
        connection.commit()
        connection.close()

    def fetch_agent_logs(self, context_id: str) -> List[Dict[str, Any]]:
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM agent_logs WHERE context_id = ? ORDER BY timestamp', (context_id,))
        rows = cursor.fetchall()
        connection.close()

        return [
            {
                'id': row[0],
                'context_id': row[1],
                'agent_name': row[2],
                'action': row[3],
                'timestamp': row[4],
                'details': json.loads(row[5])
            }
            for row in rows
        ]

    def get_agent_logs(self, context_id: str) -> List[Dict[str, Any]]:
        return self.fetch_agent_logs(context_id)


class BaseAgent:
    def __init__(self, name: str, memory: SharedMemory):
        self.name = name
        self.memory = memory
        self.logger = logging.getLogger(f"Agent.{name}")

    def log_action(self, context_id: str, action: str, details: Dict[str, Any]):
        self.memory.log_agent_action(context_id, self.name, action, details)
        self.logger.info(f"Action: {action} - {details}")

    async def process(self, data: Any, context: ContextProcessing) -> Dict[str, Any]:
        raise NotImplementedError


class ClassifierAgent(BaseAgent):
    def __init__(self, memory: SharedMemory):
        super().__init__("Classifier", memory)

    def format_detect(self, data: Any) -> DocumentType:
        if isinstance(data, dict):
            return DocumentType.JSON
        elif isinstance(data, str):
            if self._is_email_format(data):
                return DocumentType.EMAIL
            elif self._is_pdf_content(data):
                return DocumentType.PDF
        elif hasattr(data, 'read'):
            content = data.read()
            if content.startswith(b'%PDF'):
                return DocumentType.PDF
        return DocumentType.UNKNOWN

    def _is_email_format(self, text: str) -> bool:
        email_headers = ['From:', 'To:', 'Subject:', 'Date:']
        header_count = sum(1 for header in email_headers if header in text)
        if header_count >= 2:
            return True

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text) and ('From:' in text or 'To:' in text):
            return True

        return False

    def _is_pdf_content(self, text: str) -> bool:
        pdf_indicators = [
            'INVOICE', 'Bill To:', 'Invoice #:', 'Total:',
            'Payment due', 'REGULATION', 'Section', 'Effective'
        ]
        if len(text) > 100 and not self._is_email_format(text):
            indicator_count = sum(1 for indicator in pdf_indicators if indicator in text)
            if indicator_count >= 2:
                return True

            lines = text.strip().split('\n')
            if len(lines) > 5:
                return True

        return False

    def classify_intent(self, data: Any, format_type: DocumentType) -> Intent:
        text_content = self._extract_text_for_classification(data, format_type)
        text_lower = text_content.lower()

        if any(word in text_lower for word in ['invoice', 'bill', 'payment', 'amount due', 'total:']):
            return Intent.INVOICE
        elif any(word in text_lower for word in ['rfq', 'request for quote', 'quotation', 'bid']):
            return Intent.RFQ
        elif any(word in text_lower for word in ['complaint', 'issue', 'problem', 'dissatisfied']):
            return Intent.COMPLAINT
        elif any(word in text_lower for word in ['regulation', 'compliance', 'policy', 'rule']):
            return Intent.REGULATION

        return Intent.GENERAL

    def _extract_text_for_classification(self, data: Any, format_type: DocumentType) -> str:
        if format_type == DocumentType.JSON:
            return json.dumps(data) if isinstance(data, dict) else str(data)
        elif format_type == DocumentType.EMAIL or format_type == DocumentType.PDF:
            return str(data)
        return ""

    async def process(self, data: Any, context: Optional[ContextProcessing] = None) -> ContextProcessing:
        doc_format = self.format_detect(data)
        intent = self.classify_intent(data, doc_format)

        if context is None:
            context = ContextProcessing(
                id=str(uuid.uuid4()),
                source_type=doc_format,
                intent=intent,
                timestamp=datetime.now()
            )
        else:
            context.source_type = doc_format
            context.intent = intent

        context.history_processing.append(f"Classified by {self.name}: {doc_format.value} + {intent.value}")

        self.log_action(context.id, "CLASSIFY", {
            "format": doc_format.value,
            "intent": intent.value,
            "data_preview": str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
        })

        self.memory.store_context(context)
        return context


class JSONAgent(BaseAgent):
    def __init__(self, memory: SharedMemory):
        super().__init__("JSON", memory)
        self.target_schemas = {
            Intent.INVOICE: {
                "required": ["amount", "vendor", "date"],
                "optional": ["invoice_number", "description", "tax"]
            },
            Intent.RFQ: {
                "required": ["items", "quantity", "deadline"],
                "optional": ["specifications", "budget", "contact"]
            }
        }

    def validate_schema(self, data: Dict[str, Any], intent: Intent) -> Tuple[bool, List[str]]:
        if intent not in self.target_schemas:
            return True, []

        schema = self.target_schemas[intent]
        anomalies = []

        for field in schema["required"]:
            if field not in data:
                anomalies.append(f"Missing required field: {field}")

        expected_fields = set(schema["required"] + schema["optional"])
        actual_fields = set(data.keys())
        unexpected_fields = actual_fields - expected_fields

        if unexpected_fields:
            anomalies.append(f"Unexpected fields: {list(unexpected_fields)}")
        return len(anomalies) == 0, anomalies

    def extract_fields(self, data: Dict[str, Any], intent: Intent) -> Dict[str, Any]:
        if intent not in self.target_schemas:
            return data

        schema = self.target_schemas[intent]
        extracted = {}

        all_fields = schema["required"] + schema["optional"]
        for field in all_fields:
            if field in data:
                extracted[field] = data[field]

        extracted["_metadata"] = {
            "processing_timestamp": datetime.now().isoformat(),
            "schema_version": "1.0",
            "intent": intent.value
        }
        return extracted

    async def process(self, data: Dict[str, Any], context: ContextProcessing) -> Dict[str, Any]:
        is_valid, anomalies = self.validate_schema(data, context.intent)
        extracted_fields = self.extract_fields(data, context.intent)
        context.extracted_fields.update(extracted_fields)
        context.history_processing.append(f"Processed by {self.name}: extracted {len(extracted_fields)} fields")

        self.log_action(context.id, "PROCESS_JSON", {
            "is_valid": is_valid,
            "anomalies": anomalies,
            "extracted_fields_count": len(extracted_fields),
            "extracted_fields": list(extracted_fields.keys())
        })
        self.memory.store_context(context)
        return {
            "status": "success",
            "is_valid": is_valid,
            "anomalies": anomalies,
            "extracted_fields": extracted_fields,
            "context_id": context.id
        }


class EmailAgent(BaseAgent):
    def __init__(self, memory: SharedMemory):
        super().__init__("Email", memory)

    def extract_email_metadata(self, email_content: str) -> Dict[str, Any]:
        metadata = {}
        sender_match = re.search(r'From:\s*(.+)', email_content, re.IGNORECASE)
        if sender_match:
            metadata['sender'] = sender_match.group(1).strip()
        subject_match = re.search(r'Subject:\s*(.+)', email_content, re.IGNORECASE)
        if subject_match:
            metadata['subject'] = subject_match.group(1).strip()
        date_match = re.search(r'Date:\s*(.+)', email_content, re.IGNORECASE)
        if date_match:
            metadata['date'] = date_match.group(1).strip()
        to_match = re.search(r'To:\s*(.+)', email_content, re.IGNORECASE)
        if to_match:
            metadata['recipient'] = to_match.group(1).strip()

        return metadata

    def determine_urgency(self, email_content: str, subject: str = "") -> str:
        urgent_keywords = ['urgent', 'asap', 'immediate', 'emergency', 'critical', 'rush']
        high_keywords = ['important', 'priority', 'deadline', 'soon']
        content_lower = email_content.lower()
        subject_lower = subject.lower()
        if any(keyword in content_lower or keyword in subject_lower for keyword in urgent_keywords):
            return "HIGH"
        elif any(keyword in content_lower or keyword in subject_lower for keyword in high_keywords):
            return "MEDIUM"
        else:
            return "LOW"

    def extract_key_information(self, email_content: str, intent: Intent) -> Dict[str, Any]:
        info = {}
        if intent == Intent.RFQ:
            if 'quantity' in email_content.lower():
                qty_match = re.search(r'quantity[:\s]*(\d+)', email_content, re.IGNORECASE)
                if qty_match:
                    info['requested_quantity'] = int(qty_match.group(1))
            if 'deadline' in email_content.lower():
                deadline_match = re.search(r'deadline[:\s]*([^\n\r]+)', email_content, re.IGNORECASE)
                if deadline_match:
                    info['deadline'] = deadline_match.group(1).strip()
        elif intent == Intent.COMPLAINT:
            info['complaint_indicators'] = []
            complaint_words = ['dissatisfied', 'unhappy', 'problem', 'issue', 'complaint']
            for word in complaint_words:
                if word in email_content.lower():
                    info['complaint_indicators'].append(word)

        return info

    def format_for_crm(self, metadata: Dict[str, Any], urgency: str, key_info: Dict[str, Any], intent: Intent) -> Dict[
        str, Any]:
        return {
            "contact_info": {
                "sender": metadata.get('sender', 'Unknown'),
                "email_date": metadata.get('date', ''),
                "subject": metadata.get('subject', '')
            },
            "communication_details": {
                "urgency_level": urgency,
                "intent_category": intent.value,
                "key_information": key_info
            },
            "crm_metadata": {
                "processed_timestamp": datetime.now().isoformat(),
                "requires_followup": urgency in ["HIGH", "MEDIUM"],
                "auto_categorized": True
            }
        }

    async def process(self, email_content: str, context: ContextProcessing) -> Dict[str, Any]:
        metadata = self.extract_email_metadata(email_content)
        if 'sender' in metadata:
            context.sender = metadata['sender']
        if 'subject' in metadata:
            context.topic = metadata['subject']
        urgency = self.determine_urgency(email_content, metadata.get('subject', ''))
        key_info = self.extract_key_information(email_content, context.intent)
        crm_formatted = self.format_for_crm(metadata, urgency, key_info, context.intent)
        context.extracted_fields.update({
            "email_metadata": metadata,
            "urgency": urgency,
            "key_information": key_info,
            "crm_format": crm_formatted
        })
        context.history_processing.append(f"Processed by {self.name}: extracted email metadata and CRM format")
        self.log_action(context.id, "PROCESS_EMAIL", {
            "sender": metadata.get('sender', 'Unknown'),
            "urgency": urgency,
            "intent": context.intent.value,
            "key_info_fields": list(key_info.keys())
        })
        self.memory.store_context(context)
        return {
            "status": "success",
            "metadata": metadata,
            "urgency": urgency,
            "key_information": key_info,
            "crm_formatted": crm_formatted,
            "extracted_fields": {
                "email_metadata": metadata,
                "urgency": urgency,
                "key_information": key_info,
                "crm_format": crm_formatted
            },
            "context_id": context.id
        }


class PDFAgent(BaseAgent):
    def __init__(self, memory: SharedMemory):
        super().__init__("PDF", memory)

    def extract_invoice_fields(self, text: str) -> Dict[str, Any]:
        fields = {}
        amount_patterns = [
            r'total[:\s]*\$?(\d+\.?\d*)',
            r'amount[:\s]*\$?(\d+\.?\d*)',
            r'\$(\d+\.\d{2})'
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['amount'] = float(match.group(1))
                break
        inv_match = re.search(r'invoice[#\s]*:?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        if inv_match:
            fields['invoice_number'] = inv_match.group(1)
        date_patterns = [
            r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['date'] = match.group(1)
                break
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 5 and not re.match(r'^\d', line):
                fields['vendor'] = line
                break

        return fields

    def extract_regulation_fields(self, text: str) -> Dict[str, Any]:
        fields = {}
        sections = re.findall(r'section\s+(\d+\.?\d*)', text, re.IGNORECASE)
        if sections:
            fields['sections'] = sections
        effective_match = re.search(r'effective[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
        if effective_match:
            fields['effective_date'] = effective_match.group(1)

        compliance_keywords = ['must', 'shall', 'required', 'mandatory']
        requirements = []
        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in compliance_keywords):
                requirements.append(line.strip())

        if requirements:
            fields['compliance_requirements'] = requirements[:5]

        return fields

    async def process(self, pdf_text: str, context: ContextProcessing) -> Dict[str, Any]:
        extracted_fields = {}
        if context.intent == Intent.INVOICE:
            extracted_fields = self.extract_invoice_fields(pdf_text)
        elif context.intent == Intent.REGULATION:
            extracted_fields = self.extract_regulation_fields(pdf_text)
        else:
            extracted_fields = {
                "word_count": len(pdf_text.split()),
                "character_count": len(pdf_text),
                "preview": pdf_text[:200] + "..." if len(pdf_text) > 200 else pdf_text
            }

        context.extracted_fields.update({
            "pdf_content": extracted_fields,
            "text_length": len(pdf_text)
        })
        context.history_processing.append(f"Processed by {self.name}: extracted PDF fields for {context.intent.value}")

        self.log_action(context.id, "PROCESS_PDF", {
            "intent": context.intent.value,
            "text_length": len(pdf_text),
            "extracted_fields": list(extracted_fields.keys())
        })
        self.memory.store_context(context)

        return {
            "status": "success",
            "extracted_fields": extracted_fields,
            "text_length": len(pdf_text),
            "context_id": context.id
        }


class MultiAgentOrchestrator:
    def __init__(self):
        self.memory = SharedMemory()
        self.classifier = ClassifierAgent(self.memory)
        self.json_agent = JSONAgent(self.memory)
        self.email_agent = EmailAgent(self.memory)
        self.pdf_agent = PDFAgent(self.memory)
        self.logger = logging.getLogger("Orchestrator")

    async def process_input(self, data: Any) -> Dict[str, Any]:
        try:
            self.logger.info("Starting classification")
            context = await self.classifier.process(data)
            self.logger.info(f"Routing to agent for {context.source_type.value} + {context.intent.value}")

            if context.source_type == DocumentType.JSON:
                result = await self.json_agent.process(data, context)
            elif context.source_type == DocumentType.EMAIL:
                result = await self.email_agent.process(data, context)
            elif context.source_type == DocumentType.PDF:
                result = await self.pdf_agent.process(data, context)
            else:
                result = {
                    "status": "error",
                    "message": f"Unsupported format: {context.source_type.value}",
                    "context_id": context.id,
                    "extracted_fields": {}
                }
            final_result = {
                "context": asdict(context),
                "processing_result": result,
                "agent_logs": self.memory.get_agent_logs(context.id),
                "processing_complete": True
            }

            self.logger.info(f"Processing complete for context {context.id}")
            return final_result

        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "processing_complete": False,
                "processing_result": {"extracted_fields": {}}
            }

    def get_context_history(self, context_id: str) -> Optional[Dict[str, Any]]:
        context = self.memory.fetch_context(context_id)
        if context:
            return {
                "context": asdict(context),
                "agent_logs": self.memory.fetch_agent_logs(context_id)
            }
        return None

async def run_demo():
    orchestrator = MultiAgentOrchestrator()
    print("Processing JSON Invoice")
    json_invoice = {
        "invoice_number": "INV-2024-001",
        "vendor": "ACME Corp",
        "amount": 1250.00,
        "date": "2024-03-15",
        "description": "Office supplies"
    }

    result1 = await orchestrator.process_input(json_invoice)
    print(f"Result: {result1['processing_result']['status']}")
    print(f"Context ID: {result1['context']['id']}")
    print(f"Extracted Fields: {len(result1['context']['extracted_fields'])} fields\n")
    print("Processing Email RFQ")
    email_rfq = """From: john.doe@company.com
To: sales@supplier.com
Subject: RFQ - Office Furniture - URGENT
Date: Mon, 15 Mar 2024 10:30:00 +0000

Dear Sales Team,

We need to request a quote for office furniture. The requirements are:
- Quantity: 50 desk chairs
- Quantity: 25 standing desks
- Deadline: End of this month
- Budget: Up to $25,000

This is urgent as we need to furnish our new office space.

Best regards,
John Doe"""

    result2 = await orchestrator.process_input(email_rfq)
    print(f"Result: {result2['processing_result']['status']}")
    print(f"Urgency: {result2['processing_result']['urgency']}")
    print(f"Sender: {result2['context']['sender']}")
    print(f"Intent: {result2['context']['intent']}\n")
    print("Processing PDF Invoice Text")
    pdf_text = """INVOICE

ACME Corporation
123 Business St, City, State 12345

Invoice #: INV-2024-0025
Date: March 15, 2024

Bill To:
Customer Company
456 Customer Ave

Description: Consulting Services
Amount: $2,500.00
Tax: $200.00
Total: $2,700.00

Payment due within 30 days."""

    result3 = await orchestrator.process_input(pdf_text)
    print(f"Result: {result3['processing_result']['status']}")

    extracted_fields = result3['processing_result'].get('extracted_fields', {})
    print(f"Extracted Amount: ${extracted_fields.get('amount', 'N/A')}")
    print(f"Invoice Number: {extracted_fields.get('invoice_number', 'N/A')}\n")

    print("Context History for first processing:")
    history = orchestrator.get_context_history(result1['context']['id'])
    if history:
        print(f"Processing History: {history['context']['history_processing']}")
        print(f"Agent Logs: {len(history['agent_logs'])} log entries")



if __name__ == "__main__":
    asyncio.run(run_demo())
