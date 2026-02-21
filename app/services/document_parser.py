from google import genai
from google.genai import types
import os
import json
from fastapi import HTTPException
from PIL import Image
import io

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY is not configured. Please supply a valid key in the .env file."
        )
    return genai.Client()

def extract_transactions_from_document(file_content: bytes, mime_type: str):
    client = get_gemini_client()
    
    prompt = """
    You are an expert financial assistant. Extract all transaction details from this document. 
    It is either a receipt or a bank e-statement. Ignore non-transactional text.
    For each transaction found, extract the date (YYYY-MM-DD), merchant name, amount (as a float), category (e.g., Shopping, Transport, Food, Utilities, Debt), and type (income or expense).
    
    Return the result strictly as a JSON object matching this schema:
    {
      "transactions": [
        {
          "date": "2026-03-01",
          "merchant": "Example Merchant",
          "amount": 15.50,
          "category": "Food",
          "type": "expense"
        }
      ],
      "warnings": ["Any warnings or issues encountered, e.g., missing dates or blurry text"]
    }
    """
    
    try:
        if mime_type.startswith("image/"):
            # Load the image
            image = Image.open(io.BytesIO(file_content))
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
        elif mime_type == "application/pdf":
            # Pass PDF bytes directly using Part
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=file_content, mime_type="application/pdf"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

        # Parse the JSON response
        result = json.loads(response.text)
        return result
        
    except json.JSONDecodeError:
        print("Failed to parse JSON from Gemini response.")
        print(response.text)
        raise HTTPException(status_code=500, detail="Failed to parse transaction data from the document.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting data: {str(e)}")
