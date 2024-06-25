from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from crewai import Crew, Process
from tasks import fashion_task
from agents import fashion_researcher
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Forming the tech-focused crew with some enhanced configuration
crew = Crew(
    agents=[fashion_researcher],
    tasks=[fashion_task],
    process=Process.sequential,
)

# FastAPI instance
app = FastAPI()


# Define request body model
class PromptInput(BaseModel):
    prompt_line: str


# Route to handle POST requests for prompt input
@app.post("/generate-response")
def generate_response(prompt_input: PromptInput = Body(...)):
    try:
        # Starting the task execution process with enhanced feedback
        result = crew.kickoff(inputs={"prompt_line": prompt_input.prompt_line})

        # Generate response using generative AI model
        response = get_gemini_response("Convert this json format" + result)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper function to generate response using Google's generative AI
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)
    return response.text
