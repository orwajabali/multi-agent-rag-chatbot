from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response

from .utils import double_check_json_output  # Ensure this is correctly imported
from openai import OpenAI
load_dotenv()

class ClassificationAgent():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
    
    def get_response(self,messages):
        messages = deepcopy(messages)

        system_prompt = """
        You are OrwaGPT, a helpful AI academic assistant for the OrwaGPT Academy.
        Your task is to determine/analyze what agent should handle the user input. You have 6 agents to choose from:
       

        1. CourseQA_Agent: This agent is responsible for answering questions about the course content. It can provide information about lecture topics, materials, resources, content of the books, General course-related inquiries and clarifications, and more.
           EXAMPLES:  "What is the stakeholder?", "explain the difference between risk and uncertainty", "summarize all you know about system management".
        2. EmailResponse_Agent: This agent is responsible for drafting emails, and providing professional communication guidance. It can assist with writing emails to instructors or students, managing email templates, and responding to email inquiries.
           EXAMPLES: "Help me write an email to my professor", "How should I respond to this email?", "Draft an email about assignment extension".
        3. AssessmentGenerator_Agent: This agent is responsible for creating/generating new assignments, projects, quizzes, powerpoint slides/presentations, and practice exercises as user needs. It can generate questions for exams and create study materials, the user can specify the format, number and content.
           EXAMPLES: "Create a quiz about risk management", "Generate assignment questions", "Make practice problems for students", "Create a powerpoint presentation on risk mitigation".
        4. SubmissionEvaluator_Agent: This agent is responsible for evaluating and grading student work, providing feedback on assignments, projects, and exams. It can assess student performance and check answers or solutions.
           EXAMPLES: "Grade this assignment", "Provide feedback on this project", "Evaluate student responses", "Check if this answer is correct".
        5. StudentInfo_Agent: This agent is responsible for managing student enrollment information, class lists, course statistics, individual student academic records, instructor CV, and more. It can answer questions about the course syllabus, curriculum, learning objectives, and instructor and students contact information.
           EXAMPLES: "Show me the class roster", "Who are the students in this course?", "What is the registration number of?", "What do you know about this course?".
        6. Greeting_Agent: This agent is responsible for responding to user greeting messages, and small talk (e.g., "hi", "how are you", "hello", etc.).

             
            Your output should be in a structured ONLY JSON format like so. each key is a string and each value is a string. Make sure to follow the format exactly:

            {
            "chain of thought": "Go over each of the agents above and write some your thoughts about what agent is this input relevant to, and explain why you chose a specific agent based on message content.",
            "decision": "CourseQA_Agent" or "EmailResponse_Agent" or "AssessmentGenerator_Agent" or "SubmissionEvaluator_Agent" or "StudentInfo_Agent" or "Greeting_Agent". Pick one of those. and only write the word,
            "message": leave the message empty ""
            }
           


            """
        # Do not add explanation or formatting.
        #     CRITICAL REQUIREMENTS:
        #     - Choose exactly ONE agent from the five options above
        #     - Write the agent name exactly as shown (case-sensitive)
        #     - Provide thorough reasoning in "chain of thought"
        #     - Leave "message" field as empty string
        #     - Return only valid JSON with no additional text
        input_messages = [
            {"role": "system", "content": system_prompt},
        ]

        input_messages += messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
       
        output = self.postprocess(chatbot_output)
        return output
    
    def postprocess(self, output):
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            print("Primary JSON parsing failed. Attempting to correct malformed JSON...")
            corrected_output = double_check_json_output(self.client, self.model_name, output)

            if corrected_output is None:
                raise ValueError("Failed to parse and correct JSON output.")
            
            # Parse the corrected output (which is still a string)
            parsed_output = json.loads(corrected_output)

        dict_output = {
            "role": "assistant",
            "content": parsed_output['message'],
            "memory": {
                "agent": "classification_agent",
                "classification_decision": parsed_output['decision']
            }
        }
        return dict_output




