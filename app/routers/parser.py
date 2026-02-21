from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import ParserResponse
from app.services.document_parser import extract_transactions_from_document

router = APIRouter()

@router.post("/extract", response_model=ParserResponse)
async def extract_receipt_data(file: UploadFile = File(...)):
    """
    Upload a receipt image (JPEG/PNG) or bank e-statement (PDF) to automatically extract transaction details.
    Powered by Gemini Multimodal capabilities.
    """
    if not file.content_type.startswith("image/") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be an image or a PDF.")

    contents = await file.read()
    
    # Process through the document parser service
    extracted_data = extract_transactions_from_document(contents, file.content_type)
    
    # Ensure it returns the correct structure matching ParserResponse
    return ParserResponse(
        transactions=extracted_data.get("transactions", []),
        warnings=extracted_data.get("warnings", [])
    )
