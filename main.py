import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAcha9EgSQ-oX2u0MpfnxGnixA7Ds9WHoU")
genai.configure(api_key=GOOGLE_API_KEY)

# Thread pool for AI operations
executor = ThreadPoolExecutor(max_workers=4)

# FastAPI App
app = FastAPI(
    title="Medical Codes Chatbot API with Gemini AI",
    description="Chatbot with deep trip analysis, medical code Q&A, and searchable knowledge",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility function to run Gemini AI call async with threadpool
async def generate_gemini_response(prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            executor,
            lambda: genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,  # Lower temperature for more precise, factual responses
                    max_output_tokens=max_tokens,
                    top_p=0.85,  # Slightly increased for better quality
                    top_k=40
                )
            )
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation error: {str(e)}")

# Request and Response Models
class TripAnalysisRequest(BaseModel):
    trip_data: Dict[str, Any]

class ChatQueryRequest(BaseModel):
    question: str

# =====================================
# 1. Trip Deep Analysis Endpoint - IMPROVED PROMPTS ONLY
# =====================================
@app.post("/api/trip/analyze")
async def trip_deep_analysis(request: TripAnalysisRequest):
    """Enhanced with better prompting - same response format"""
    
    # IMPROVED: More specific, structured prompt with role definition and examples
    system_prompt = f"""You are a certified medical billing analyst with 15+ years of experience in healthcare revenue cycle management. Your expertise includes ICD-10, CPT, HCPCS coding, claim denial analysis, and compliance auditing.

TASK: Perform a comprehensive trip analysis on the provided medical billing data. Provide actionable, specific insights with exact numbers and code references.

TRIP DATA TO ANALYZE:
{json.dumps(request.trip_data, indent=2)}

ANALYSIS REQUIREMENTS:
1. **Compliance Review**: Check all medical codes (ICD-10, CPT, HCPCS) for accuracy, proper pairing, and documentation requirements. Cite specific code numbers.

2. **Risk Assessment**: Identify denial risks with probability estimates. List specific problematic elements (e.g., "CPT 99214 lacks supporting documentation for level 4 visit").

3. **Financial Impact**: Calculate exact dollar amounts at risk. Example: "Claim #12345 risks $450 denial due to unbundling violation."

4. **Optimization Opportunities**: Suggest specific coding improvements to maximize reimbursement. Example: "Consider upcoding from 99213 to 99214 if documentation supports 30+ minutes."

5. **Action Items**: Provide numbered, prioritized list of immediate actions with deadlines.

RESPONSE RULES:
- Be specific: Always include code numbers, dollar amounts, claim IDs
- Be concise: Each section should be 2-4 sentences maximum
- Be actionable: Every recommendation must be implementable
- Be accurate: Only state facts verifiable from the provided data

ANALYSIS:"""

    ai_response = await generate_gemini_response(system_prompt, max_tokens=1500, temperature=0.1)
    
    # SAME response format - no changes
    return {"trip_analysis": ai_response}

# =====================================
# 2. Medical Code Chatbot Query Endpoint - IMPROVED PROMPTS ONLY
# =====================================
@app.post("/api/chat/medical-code")
async def medical_code_chat(request: ChatQueryRequest):
    """Enhanced with better prompting - same response format"""
    
    # IMPROVED: Detailed role definition with few-shot examples for accuracy
    system_prompt = f"""You are MediBill AI, a certified professional coder (CPC) and medical billing expert. You have deep expertise in:
- ICD-10-CM diagnosis codes (73,000+ codes)
- CPT procedure codes (10,000+ codes)
- HCPCS Level II codes
- Medicare/Medicaid/Commercial payer rules
- CMS billing guidelines
- Medical necessity criteria
- Modifier usage and bundling rules

INSTRUCTION: Answer the user's question with precise, accurate information. Always include:
- Specific code numbers when applicable
- Exact billing rules or regulations
- Payer-specific requirements if relevant
- Step-by-step guidance for complex questions

FEW-SHOT EXAMPLES FOR ACCURACY:

Example 1:
Q: What is CPT 99213?
A: CPT 99213 is an Evaluation and Management code for an established patient office visit of low to moderate complexity. Requirements: (1) Problem-focused history, (2) Problem-focused exam, (3) Low complexity medical decision making. Typical time: 15 minutes. Reimbursement: Medicare pays approximately $93-$110 depending on locality.

Example 2:
Q: Can I bill CPT 99214 and 96372 together?
A: Yes, CPT 99214 (office visit) and 96372 (therapeutic injection) can be billed together for the same date of service. However, you must append modifier 25 to the E/M code (99214-25) to indicate a separately identifiable service. Documentation must support both services.

Example 3:
Q: ICD-10 for Type 2 diabetes with neuropathy?
A: Use E11.40 (Type 2 diabetes mellitus with diabetic neuropathy, unspecified). If the neuropathy type is specified, use more specific codes: E11.41 (mononeuropathy), E11.42 (polyneuropathy), E11.43 (autonomic neuropathy), or E11.44 (amyotrophy).

USER QUESTION:
{request.question}

ANSWER (be specific with code numbers, rules, and guidelines):"""

    ai_response = await generate_gemini_response(system_prompt, max_tokens=1000, temperature=0.1)
    
    # SAME response format - no changes
    return {"response": ai_response}

# =====================================
# 3. Trip-Related Question & Answer Endpoint - IMPROVED PROMPTS ONLY
# =====================================
@app.post("/api/chat/trip-question")
async def trip_question_chat(request: Dict[str, Any]):
    """Enhanced with better prompting - same response format"""
    
    trip_context = request.get("trip_context", {})
    question = request.get("question", "")
    
    if not trip_context or not question:
        raise HTTPException(status_code=400, detail="trip_context and question are required")

    context_str = json.dumps(trip_context, indent=2)
    
    # IMPROVED: Context-aware prompt with explicit grounding instructions
    system_prompt = f"""You are a medical billing AI analyst with complete knowledge of the trip context provided below. Answer questions ONLY based on the specific data in this trip.

CRITICAL INSTRUCTIONS:
1. **Ground all answers in the trip data**: Reference specific values, claim IDs, codes, or amounts from the context
2. **Be precise**: If the trip shows "$450.00", say "$450.00" not "approximately $450"
3. **Cite specifics**: Always mention which claim, line item, or field you're referencing
4. **Admit limitations**: If information is not in the trip data, clearly state "This information is not present in the current trip data"
5. **Calculate accurately**: When doing math (totals, percentages), show your work

TRIP CONTEXT:
{context_str}

EXAMPLE RESPONSES FOR REFERENCE:

Q: What is the total claim amount?
Good: "The total claim amount is $1,247.50, calculated from claim lines: $450.00 (CPT 99214) + $325.00 (CPT 93000) + $472.50 (CPT 80053)."
Bad: "The total is around $1,200."

Q: Are there any coding errors?
Good: "Yes, Line 3 shows CPT 80053 paired with ICD-10 Z00.00 (routine health exam). This is a medical necessity mismatch - lab panels require a specific diagnosis, not a screening code."
Bad: "There might be some issues with the codes."

USER QUESTION:
{question}

ANSWER (reference specific data from the trip context):"""

    ai_response = await generate_gemini_response(system_prompt, max_tokens=1000, temperature=0.1)
    
    # SAME response format - no changes
    return {"response": ai_response}

# =====================================
# 4. General Search Endpoint - IMPROVED PROMPTS ONLY
# =====================================
@app.get("/api/search")
async def general_search(query: str = Query(..., min_length=1)):
    """Enhanced with better prompting - same response format"""
    
    # IMPROVED: Structured search prompt with specific formatting
    system_prompt = f"""You are MediBill Search AI, a knowledgeable medical billing assistant. Provide concise, authoritative answers to search queries.

SEARCH QUERY: "{query}"

RESPONSE FORMAT:
1. **Direct Answer** (1-2 sentences): The core answer to the query
2. **Key Details** (2-3 bullet points): Specific codes, numbers, rules, or facts
3. **Related Concepts** (optional): Briefly mention 1-2 related topics if helpful

ACCURACY RULES:
- Always include specific code numbers (e.g., "CPT 99213", not "office visit code")
- Include exact dollar amounts when discussing fees
- Cite payer rules if applicable (Medicare, Medicaid, commercial)
- If uncertain, state "Consult payer-specific guidelines" rather than guessing

EXAMPLE RESPONSES:

Search: "CPT modifier 25"
Answer: 
**Direct Answer**: Modifier 25 indicates a separately identifiable E/M service on the same day as a procedure.
**Key Details**:
- Required when billing E/M code with minor procedure same day
- Documentation must support distinct services
- Common use: 99214-25 with 11200 (skin lesion removal)
**Related**: See also modifier 57 for major procedures

Search: "denial code CO-16"
Answer:
**Direct Answer**: CO-16 means "Claim lacks information needed for adjudication" - the claim is missing required data.
**Key Details**:
- Common causes: Missing diagnosis code, invalid procedure code, or incomplete patient info
- Resolution: Review claim for blank fields or invalid entries
- Resubmit with corrected information within timely filing limits
**Related**: See also CO-4 (procedure code inconsistent with modifier)

YOUR SEARCH RESPONSE FOR: "{query}"
"""

    ai_response = await generate_gemini_response(system_prompt, max_tokens=500, temperature=0.15)
    
    # SAME response format - no changes
    return {"search_response": ai_response}

# =====================================
# 5. Health Check Endpoint - SAME
# =====================================
@app.get("/health")
async def health_check():
    try:
        test_prompt = "Respond with exactly 'OK' if operational."
        response = await generate_gemini_response(test_prompt, max_tokens=10, temperature=0.0)
        health_status = "operational" if response and "OK" in response else "degraded"
    except Exception:
        health_status = "offline"
    return {"status": "healthy", "ai_status": health_status}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
