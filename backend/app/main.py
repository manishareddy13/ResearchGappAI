
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from io import BytesIO

app = FastAPI(
    title="ResearchGap AI",
    description="AI-powered research paper analysis platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ResearchGap AI",
        "status": "Application is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        content = await file.read()

        reader = PdfReader(BytesIO(content))

        extracted_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF"
            )

        return {
            "filename": file.filename,
            "total_pages": len(reader.pages),
            "extracted_text": extracted_text
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(error)}"
        )
