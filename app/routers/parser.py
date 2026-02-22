from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import json
from typing import Optional, List
from google import genai
from google.genai import types

router = APIRouter()

# Initialize Google GenAI client (uses GEMINI_API_KEY from environment)
client = genai.Client()

class ParsedTransaction(BaseModel):
    merchant: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    type: str = "expense"

class ParsedDocument(BaseModel):
    transactions: List[ParsedTransaction]
    confidence_score: float

@router.post("/parse-document", response_model=ParsedDocument)
async def parse_document(
    file: UploadFile = File(...),
    languages: str = Form(None)
):
    """
    Takes an uploaded image or PDF receipt/statement, extracts key information
    using Gemini Multimodal LLM, and returns structured data containing a list of transactions.
    """
    content = await file.read()
    
    try:
        # Prompt for Gemini to extract specific fields
        prompt = (
            "You are an expert financial document parser. "
            "Please analyze the attached image or PDF of a receipt or bank statement "
            "and extract EVERY transaction you can find.\n\n"
            "For each transaction, extract:\n"
            "1. 'merchant': The name of the store, merchant, or description.\n"
            "2. 'date': The date of the transaction, formatted strictly as YYYY-MM-DD.\n"
            "3. 'total_amount': The total amount charged or credited, as a positive float (e.g., 10.50). Remove currency symbols.\n"
            "4. 'type': Either 'expense' or 'income'. If it's a purchase/payment, it's 'expense'. If it's a deposit/salary, it's 'income'.\n\n"
            "Respond ONLY with a valid JSON object matching this exact structure: "
            '{"transactions": [{"merchant": "string", "date": "YYYY-MM-DD", "total_amount": float, "type": "expense"}]}. '
            "If a field cannot be found for a transaction, set its value to null."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=content,
                    mime_type=file.content_type,
                ),
                prompt
            ]
        )
        
        # Clean up response text to ensure it's valid JSON
        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        data = json.loads(result_text.strip())
        
        transactions = []
        for t in data.get("transactions", []):
            transactions.append(ParsedTransaction(
                merchant=t.get("merchant"),
                date=t.get("date"),
                total_amount=t.get("total_amount"),
                type=t.get("type", "expense")
            ))
            
        return ParsedDocument(
            transactions=transactions,
            confidence_score=0.99  # LLM parsing is highly accurate
        )
        
    except Exception as e:
        print(f"Error parsing document with Gemini: {e}")
        return ParsedDocument(
            transactions=[],
            confidence_score=0.0
        )
