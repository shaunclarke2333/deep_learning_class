import json                                              
from typing import Any
# Hugging face Inference Client for making API calls to Hugging Face models                              
from huggingface_hub import InferenceClient

#Definingin variables that will be used throughout the code
file_path: str = "user_memory.json"  # Path to the JSON file where user memory will be stored
memory_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # Model used for generating responses based on memory
max_tokens: int = 150  # Maximum number of tokens to generate in the response
temperature: float = 0.3  # Temperature for controlling the randomness of the generated response

# This method returns the complete memory database, or empty dict if no file exists yet.
def load_memory() -> dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
# This method accepts raw feedback, summarizes it via the hugging face model, and stores the summary in the memory database.
def update_user_memory(user_name: str, new_feedback: str, hf_token: str) -> None:
    # Load existing memory
    try:
        with open(file_path, "r") as f:
            database: dict[str, Any] = json.load(f)
    except FileNotFoundError:
        database: dict[str, Any] = {}

    # Preparing the summarization LLM call.
    # We ask the LLM to act as a "Data Clerk" to keep the memory clean.
    previous_context: str = database.get(user_name, "No previous data.")

    prompt: str = f"""
    You are a memory management assistant.
    Current Memory for {user_name}: {previous_context}
    New User Feedback: {new_feedback}

    Task: Create a concise, one-sentence summary of the user's permanent preferences
    based on the new feedback and the old memory only. Return ONLY the summary.
    """

    # The memory LLM call
    # Instantioating the Hugging Face Inference Client with the specified model and token
    client: InferenceClient = InferenceClient(model=memory_model, token=hf_token)
    # making the API call to generate a summary based on the prompt, with specified max tokens and temperature
    response = client.chat_completion(
        messages=[{"role": "system", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Extracting the summary from the response
    summarized_memory: str = response.choices[0].message.content.strip()

    # Update the database with the new summary for the user
    database[user_name] = summarized_memory

    # Save the updated database back to the JSON file
    with open(file_path, "w") as f:
        json.dump(database, f, indent=4)

    print(f"Memory updated for {user_name}!")
    return summarized_memory
