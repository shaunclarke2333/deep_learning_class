"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Assignment: week 6 - Generative AI Language Models
"""
# This allows us to hide the Hugging Face token as we type it in
import getpass                                           
from agent import generate_menu                          
from memory import load_memory, update_user_memory 

# This prevents us from looping foreever anf alling into the silent success trap
max_feedback_rounds: int = 5
approval_words: tuple[str] = ("yes", "yep", "y", "approve", "approved", "looks good", "perfect", "great", "love it", "that's it")

def main() -> None:
    
    # Get the Hugging Face token from the user securely
    hf_token: str = getpass.getpass("Please enter your Hugging Face API token: ")

    # onboard the user #
    # Get the user's name and preferences
    user_name: str = input("What is your name? ").strip()

    # Loading database memory to get the user's previous preferences,
    # if they exist. This allows us to personalize the menu even on the first round of feedback.
    database = load_memory()

    # If the user has previous preferences, we load them. If not, we start with a blank slate.
    learned_prefs: str = database.get(user_name, "")

    # Checking if user is new to decide on how to handle first interaction
    if not learned_prefs:
        print(f"Welcome, {user_name}! It looks like this is your first time using the menu generator. Let's start by getting to know your food preferences.")
        profile = input("Please tell me about your dietary restrictions, favorite cuisines, and any specific likes or dislikes you have when it comes to food: ")
    else:
        print(f"Welcome back, {user_name}! Based on your previous interactions, I have the following preferences on file for you: {learned_prefs}")
        update = input("Would you like to update your preferences? (yes/no) ").lower()
        if update in ("yes", "y", "update"):
            profile = input("Please provide any updates to your dietary restrictions, favorite cuisines, or specific likes/dislikes: ")       
        else:
            profile = ""

    session_feedback: str = ""

    # Loop to get feedback and update memory until the user approves the menu or we reach the max feedback rounds
    for round_num in range(max_feedback_rounds):
        # generating the menu
        menu = generate_menu(profile, learned_prefs, session_feedback, hf_token)
        print(menu)

        # Getting user response
        user_response = input("Approve? (yes / type feedback): ").strip().lower()

        if user_response in approval_words:
            break
        else:
            session_feedback = user_response
            learned_prefs = update_user_memory(user_name, session_feedback, hf_token)

    else:
        # Failing loudly if the for loop exits without break. This means we hit the max rounds thershold
        raise RuntimeError("Max feedback rounds reached without approval.")
    

    print("The waiter will be right over to take your order.")


if __name__ == "__main__":
    main()








