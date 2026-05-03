"""
generator.py — Build the RAG prompt and call the Gemini LLM to generate
a grounded, hallucination-free response from retrieved product context.
"""
 
import os
import google.generativeai as genai
from dotenv import load_dotenv
 
load_dotenv()   # Load GEMINI_API_KEY from .env file
api_key = os.getenv("GEMINI_API_KEY") 
 
# ── Configuration ──────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-2.5-flash-lite"
MAX_TOKENS     = 1024
TEMPERATURE    = 0.2   # Low temperature = factual, deterministic answers
 
SYSTEM_PROMPT = """You are a helpful retail customer support assistant for an online store.
Your ONLY job is to answer customer questions about products using the product information provided below.
 
STRICT RULES:
1. Answer ONLY from the CONTEXT provided. Do not use any external knowledge.
2. If the answer is not in the context, respond EXACTLY with:
   "I don't have that information in our product catalogue. Please contact support at support@store.com."
3. Be concise, friendly, and professional.
4. For price-related queries, always include the exact price from the context.
5. For comparisons, list differences clearly in bullet points.
6. Never make up product features, prices, or policies.
"""
 
 
def configure_gemini():
    """Configure the Gemini API using the key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=your_key."
        )
    genai.configure(api_key=api_key)
 
 
def build_prompt(retrieved_chunks: list[dict], user_question: str) -> str:
    """
    Construct the full RAG prompt from retrieved context + user question.
 
    Args:
        retrieved_chunks: List of chunk dicts from the Retriever.
        user_question: The customer's original question.
 
    Returns:
        A formatted prompt string to send to the LLM.
    """
    if not retrieved_chunks:
        context_text = "No relevant product information found."
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"--- Product Result {i} (Source: {chunk['source']}) ---")
            context_parts.append(chunk["text"])
        context_text = "\n\n".join(context_parts)
 
    prompt = f"""
CONTEXT (Retrieved Product Information):
================================
{context_text}
================================
 
CUSTOMER QUESTION: {user_question}
 
ANSWER:"""
    return prompt.strip()
 
 
def generate_response(retrieved_chunks: list[dict], user_question: str) -> str:
    """
    Main function: build the prompt, call Gemini, and return the answer.
 
    Args:
        retrieved_chunks: Retrieved product context from the vector store.
        user_question: The customer's question.
 
    Returns:
        A grounded natural language answer string.
    """
    try:
        configure_gemini()
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            )
        )
 
        full_prompt = build_prompt(retrieved_chunks, user_question)
        response = model.generate_content(full_prompt)
        return response.text.strip()
 
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try again or contact support."
