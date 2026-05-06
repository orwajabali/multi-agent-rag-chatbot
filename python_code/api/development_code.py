from agents import (GuardAgent,
                    ClassificationAgent,
                    QACourseAgent,
                    AgentProtocol,
                    AssessmentGeneratorAgent,
                    StudentInfoAgent,
                    EmailResponseAgent,
                    GreetingAgent,
                    SubmissionEvaluatorAgent) 
import os
from typing import Dict
import re
import tempfile
import shutil

def extract_file_paths_from_message(message):
    """Extract file paths from a message that mentions files."""
    # This is a simple regex pattern to find file paths
    # In a real implementation, you might need more sophisticated parsing
    file_pattern = r'file:([^\s]+)'
    matches = re.findall(file_pattern, message)
    return matches

def handle_file_upload(file_path, file_type):
    """Handle file upload and return the temporary path."""
    # In a real implementation, this would handle the file upload process
    # For this example, we'll just return the path as if it was uploaded
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, os.path.basename(file_path))
    
    # Copy the file to the temporary location (simulating upload)
    try:
        shutil.copy(file_path, temp_path)
        print(f"File '{file_path}' uploaded as {file_type} to {temp_path}")
        return temp_path
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def main():
    pass

if __name__ == "__main__":
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()

    agent_dict: Dict[str, AgentProtocol] = {
        "CourseQA_Agent": QACourseAgent(),
        "EmailResponse_Agent": EmailResponseAgent(),
        "AssessmentGenerator_Agent": AssessmentGeneratorAgent(),
        "SubmissionEvaluator_Agent": SubmissionEvaluatorAgent(),
        "StudentInfo_Agent": StudentInfoAgent(),
        "Greeting_Agent": GreetingAgent()
    }
    
    # Initialize submission files dictionary
    submission_files = {
        "rubric": None,
        "assignment_details": None,
        "submission": None
    }
    
    messages = []
    while True:
        # Display the chat history
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        
        # Check if the user is uploading files
        if prompt.lower().startswith("upload"):
            parts = prompt.split()
            if len(parts) >= 3:
                file_type = parts[1].lower()  # rubric, assignment_details, or submission
                file_path = parts[2]
                
                if file_type in submission_files:
                    temp_path = handle_file_upload(file_path, file_type)
                    if temp_path:
                        submission_files[file_type] = temp_path
                        prompt = f"I've uploaded a {file_type} file: {file_path}"
                    else:
                        prompt = f"Failed to upload {file_type} file: {file_path}"
                else:
                    prompt = f"Unknown file type: {file_type}. Please use 'rubric', 'assignment_details', or 'submission'."
        
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent = classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)

        # Special handling for SubmissionEvaluator_Agent
        if chosen_agent == "SubmissionEvaluator_Agent":
            # Check if all required files are uploaded
            missing_files = [file_type for file_type, path in submission_files.items() if path is None]
            
            if missing_files:
                # If files are missing, inform the user
                missing_files_str = ", ".join(missing_files)
                response = {
                    "role": "assistant",
                    "content": f"I need the following files to evaluate the submission: {missing_files_str}. Please upload them using 'upload [file_type] [file_path]'.",
                    "memory": {}
                }
                messages.append(response)
                continue
            
            # If all files are available, prepare the agent with the files
            agent = agent_dict[chosen_agent]
            
            # Load the files into the SubmissionEvaluatorAgent
            agent.load_rubric(submission_files["rubric"], "text")
            agent.load_assignment_details(submission_files["assignment_details"], "text")
            agent.load_submission(submission_files["submission"])
            
            # Evaluate the submission
            agent.evaluate_submission()
            
            # Get the evaluation results
            evaluation_results = agent.get_evaluation()
            
            # Create a response with the evaluation results
            response = {
                "role": "assistant",
                "content": f"Evaluation Results:\n\n{evaluation_results}",
                "memory": {"evaluation_results": evaluation_results}
            }
        else:
            # For other agents, proceed as normal
            agent = agent_dict[chosen_agent]
            response = agent.get_response(messages)
        
        # Add the response to messages
        messages.append(response)

print("""
Usage Instructions:

1. To upload files for submission evaluation:
   - upload rubric /path/to/rubric.txt
   - upload assignment_details /path/to/assignment_details.txt
   - upload submission /path/to/submission.pdf

2. To request an evaluation, ask something like:
   "Please evaluate my submission based on the rubric"

3. The system will check if all required files are uploaded before proceeding with evaluation.

4. If any files are missing, you'll be prompted to upload them.
""")
