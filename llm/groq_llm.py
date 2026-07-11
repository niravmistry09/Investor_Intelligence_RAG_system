import os
import re
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def get_groq_client() -> Groq:
    """Initialize and return the Groq client."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# For testing script execution directly
if __name__ == "__main__":
    client = get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    print(chat_completion.choices[0].message.content)


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:
    """
    Generate structured output using Groq Cloud.

    Args:
        prompt: Input prompt.
        response_model: Pydantic response model.
        model: Groq model name.

    Returns:
        Parsed response model.
    """
    # Read model name from environment when not provided; fallback to a good default if empty
    model = model or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

    # Use the Groq client instead of OpenAI
    client = get_groq_client()

    messages = [
        {
            "role": "system",
            "content": "You are an expert financial analyst."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Prefer structured parsing when supported by the deployment
    try:
        # Groq also supports the .beta.chat.completions.parse syntax for Pydantic models!
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model
        )

        # Debug: show structured parsed output
        try:
            parsed = response.choices[0].message.parsed
            print("[debug] Structured parsed output:", parsed)
        except Exception:
            print("[debug] Structured response received but could not access parsed field")

        return response.choices[0].message.parsed

    except Exception as exc:
        # Fallback: Catching generic exceptions since we shifted from openai.BadRequestError
        msg = str(exc)
        
        # Check if Groq failed because of structured formats or if we need a standard JSON fallback
        fallback = client.chat.completions.create(
            model=model,
            messages=messages,
            # We can also pass json mode explicitly to Groq if the schema parsing fails
            response_format={"type": "json_object"} if "json" in msg.lower() else None
        )

        text = fallback.choices[0].message.content
        print("[debug] Fallback raw text response:\n", text)

        # Try to extract the first JSON object from the model output (Original developer's glue code)
        match = re.search(r"\{.*\}", text, re.S)
        json_text = match.group(0) if match else text

        try:
            # Handle explicit 'null' responses: return an empty model instance
            if isinstance(json_text, str) and json_text.strip() in ("null", "None", ""):
                if hasattr(response_model, "model_construct"):
                    return response_model.model_construct()
                try:
                    return response_model.model_validate({}) if hasattr(response_model, "model_validate") else response_model.parse_obj({})
                except Exception:
                    raise RuntimeError("Model returned null and cannot construct an empty instance.")

            # pydantic validation (Kept exactly as the original code)
            if hasattr(response_model, "model_validate_json"):
                parsed = response_model.model_validate_json(json_text)
            else:
                parsed = response_model.parse_raw(json_text)

            return parsed
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON fallback response: {e}\nRaw output:\n{text}") from e