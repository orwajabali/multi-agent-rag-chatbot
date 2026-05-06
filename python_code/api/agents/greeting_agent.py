# -*- coding: utf-8 -*-
"""
This module defines the GuardAgent class.
"""
import os
import json
import re
# Removed deepcopy as it's handled by the controller if needed
# from copy import deepcopy 
from .utils import get_chatbot_response, double_check_json_output
from openai import OpenAI
# Removed load_dotenv as it should be called once at the application entry point
# from dotenv import load_dotenv
# load_dotenv()

# Removed OpenAI import as client is passed in
# from openai import OpenAI 

class GreetingAgent():
    """
    Agent responsible for checking if a message is relevant to the course.
    Accepts shared OpenAI client for efficiency.
    """
    def __init__(self):
        """Initializes the GuardAgent with a shared API client and model name."""
        # Use the passed-in client and model name
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        ) 
        self.model_name = os.getenv("MODEL_NAME")
    
    def get_response(self, messages):
        """
        Classifies a message as ALLOWED or NOT ALLOWED based on course relevance.

        Args:
            messages (List[Dict[str, Any]]): Conversation history.

        Returns:
            Dict[str, Any]: Agent's response including the decision.
        """
        # No need to deepcopy messages here if the agent doesn't modify it before the API call
        # messages = deepcopy(messages) 

        system_prompt = """
                You are a smart academic assistant for the Smart System Management course. Your job is to determine whether a student's message is relevant to this course and should be answered by the course-specific AI assistant.

                Your task is to answer greeting messages (e.g., hi, how are you , hello, ect).

                Respond ONLY with valid JSON in **one of these two formats** (no extra text):

                If the message **is allowed**:

                {
                "chain of thought": "Explain why the message is allowed based on its relevance to the Smart System Management course.",
                "decision": "allowed",
                "message": "Hello! How can I assist you today?"
                }

                If the message **is not allowed**:

                {
                "chain of thought": "Explain why the message is not allowed. It is not relevant to the Smart System Management course.",
                "decision": "not allowed",
                "message": "Sorry, I can only help with questions related to the Smart System Management course. Please contact your instructor or university support for other matters."
                }
                """

        # Select relevant parts of the message history (e.g., last message)
        # Using last message for guard check is usually sufficient and efficient
        input_messages = [
            {"role": "system", "content": system_prompt},
            messages[-1] # Pass only the last user message
        ]

        # Use the shared client instance passed during initialization
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)

        
        # double check json 
        chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
                
        # Postprocessing now includes JSON parsing and error handling
        output = self.postprocess(chatbot_output)

        return output

    def postprocess(self, chatbot_output):
        """
        Processes the raw output from the chatbot, parsing JSON and handling errors.

        Args:
            chatbot_output (str): The raw string response from the chatbot.

        Returns:
            Dict[str, Any]: A dictionary containing the agent's response and memory.
        """
        try:
            # print("DEBUG chatbot_output:", repr(chatbot_output)) 
            # Attempt to parse the JSON string directly
            output_data = json.loads(chatbot_output)
            
            # Basic validation for required keys
            if not all(k in output_data for k in ["decision", "message"]):
                 raise ValueError("Missing required keys in guard agent output")

            # Structure the final response
            dict_output = {
                "role": "assistant",
                "content": output_data.get("message", ""), # Use message from JSON
                "memory": {
                    "agent": "Greeting_Agent",
                    "guard_decision": output_data.get("decision", "not allowed"), # Default to not allowed on error
                    "chain_of_thought": output_data.get("chain_of_thought", "")
                }
            }
        except json.JSONDecodeError:
            # Handle cases where the output is not valid JSON
            print(f"Error: Greeting_Agent failed to decode JSON: {chatbot_output}")
            # Return a default 'not allowed' response for safety
            dict_output = {
                "role": "assistant",
                "content": "Sorry, I encountered an internal error. Please try again.", # Generic error
                "memory": {"agent": "Greeting_Agent", "guard_decision": "not allowed", "error": "JSONDecodeError"}
            }
        except ValueError as ve:
            # Handle cases where JSON is valid but missing keys
            print(f"Error: Greeting_Agent validation error: {ve}")
            dict_output = {
                "role": "assistant",
                "content": "Sorry, I encountered an internal error processing the request structure.",
                "memory": {"agent": "Greeting_Agent", "guard_decision": "not allowed", "error": "ValueError", "details": str(ve)}
            }
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Error: Unexpected error in GuardAgent postprocessing: {e}")
            dict_output = {
                "role": "assistant",
                "content": "An unexpected error occurred. Please try again later.",
                "memory": {"agent": "Greeting_Agent", "guard_decision": "not allowed", "error": "UnexpectedError", "details": str(e)}
            }
            
        return dict_output

