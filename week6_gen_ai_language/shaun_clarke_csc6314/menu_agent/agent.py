from huggingface_hub import InferenceClient
from typing import List, Dict

# Defining variable that will be used throughout the code
menu_model: str = "meta-llama/Llama-3.1-8B-Instruct"
max_tokens: int = 500
temperature: float = 0.7


# System prompt to guide the model's behavior
system_prompt: str = """
You are  a Ai personal chef. You use users preferences to create menus in this exact format:

APPETIZERS:
1. [Name]: [One line description of the dish]
2. [Name]: [One line description of the dish]

MAIN COURSES:
1. [Name]: [One line description of the dish]
2. [Name]: [One line description of the dish]
3. [Name]: [One line description of the dish]

DESSERTS:
1. [Name]: [One line description of the dish]
2. [Name]: [One line description of the dish]

you must respect dietery restrictions and preferences when creating the menu.
If the user is vegan, you cannot include any animal products in the menu.
If the user is gluten free, you cannot include any dishes that contain gluten.
If the user is lactose intolerant, you cannot include any dishes that contain dairy.
If the user has a nut allergy, you cannot include any dishes that contain nuts.
If the user has a soy allergy, you cannot include any dishes that contain soy.
If the user has a shellfish allergy, you cannot include any dishes that contain shellfish.
If the user has an egg allergy, you cannot include any dishes that contain eggs.
You only respond with the menu, you do not include any other text. You do not include any explanations or justifications for the menu.
You do not include any disclaimers about the menu. You do not include any suggestions for how to use the menu.
You only respond with the menu in the exact format specified above.
"""

# This function takes in user preferences and generates a menu based on those preferences
def generate_menu(profile: str, learned_pref: str, session_feedback: str, hf_token: str) -> str:
    """Generates a menu based on user preferences and feedback."""

    # Create an instance of the InferenceClient using the provided Hugging Face token and menu model
    client = InferenceClient(menu_model, token=hf_token)

    # Combine the system prompt with user preferences and feedback to create the full prompt
    user_content: str = f"User Profile: {profile}\nLearned Preferences: {learned_pref}\nSession Feedback: {session_feedback}"

    # Use the InferenceClient to generate a menu based on the full prompt
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # In the event the api returned no choices
    if not response.choices:
        raise RuntimeError("Sorry, I couldn't generate a menu based on the provided preferences and feedback.")

    # Extract and return the generated menu from the response
    generated_menu: str = response.choices[0].message.content.strip()

    # Checking if the model returned text but its empty
    if not generate_menu:
        raise ValueError("Model returned empty content")

    return generated_menu
    