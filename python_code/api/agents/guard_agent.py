# -*- coding: utf-8 -*-
"""
This module defines the GuardAgent class.
"""
import os
import json
import re

from .utils import get_chatbot_response, double_check_json_output
from openai import OpenAI


class GuardAgent():
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
        self.instructor_name = "Dr. Allam Mousa"  # Example instructor name, can be made dynamic if needed
        self.course_name = "Smart System Management"  # Example course name, can be made dynamic if needed

        system_prompt = f"""
                You are OrwaGPT, a smart academic assistant for the {self.course_name} course.
                Your task is to determine whether the user is asking something relevant to the {self.course_name} course or not.
                IMPORTANT: Any mention of "the course", "this course", or general course-related language should ALWAYS be interpreted as referring to the **{self.course_name} course**, even if the full name is not used.
                You always assume that user messages are referring to the {self.course_name} course unless they explicitly mention another context.

                The user is allowed to:
                1. Ask questions about the course, like course content, assignments, exams, syllabus, grading, learning objectives, schedule, or topics.
                2. Ask questions about students, enrollment/registration information like registration number, email address, status or course-related student queries.
                3. Ask questions about course materials, such as textbooks, lecture notes, or supplementary resources.
                4. Ask questions about {self.instructor_name}, such as their contact information, office hours, CV, or teaching assistants.
                5. Ask questions to generate/create assessments, quizzes, exams, projects, papers, powerpoint slides/presentations, or problems related to the course.
                6. Ask questions to evaluate submissions, assignments, or projects related to the course.
                7. Ask questions to generate emails related to the course, such as emails to instructors, students, or teaching assistants.
                8. Greeting the OrwaGPT assistant, ask how are you, such as "Hi OrwaGPT", "Hello OrwaGPT", or "OrwaGPT, how are you?".

                The user is NOT allowed to:

                1. Ask questions about other courses.

                Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
                {{
                "chain of thought": "go over each of the points above and make see if the message lies under this point or not. Then you write some your thoughts about what point is this input relevant to.",
                "decision": "allowed" or "not allowed". Pick one of those. and only write the word,
                "message": "" if it's allowed, otherwise write "Sorry, This assistant is specialized for Smart System Management course support only. For other matters, please refer to the appropriate university resources or departments."
                }}
                
                
                """

        input_messages = [
            {"role": "system", "content": system_prompt},
            messages[-1] # Pass only the last user message
        ]

        # Use the shared client instance passed during initialization
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)

        # # double check json 
        # chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
                
        # Postprocessing now includes JSON parsing and error handling
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self, output):
        try:
            output_json = json.loads(output)
            dict_output = {
                "role": "assistant",
                "content": output_json.get('message', ''),
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": output_json.get('decision', 'not allowed')
                }
            }
        except json.JSONDecodeError:
            # Handle plain text output (fallback)
            print("⚠️ Warning: Output is not valid JSON. Wrapping it manually.")
            dict_output = {
                "role": "assistant",
                "content": output,
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": "allowed"  # or use 'not allowed' depending on your use case
                }
            }

        return dict_output


    # def postprocess(self,output):
    #     output = json.loads(output)

    #     dict_output = {
    #         "role": "assistant",
    #         "content": output['message'],
    #         "memory": {"agent":"guard_agent",
    #                    "guard_decision": output['decision']
    #                   }
    #     }
    #     return dict_output

