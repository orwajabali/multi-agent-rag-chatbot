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
import pathlib
import tempfile
import shutil
import base64

folder_path = pathlib.Path(__file__).parent.resolve()

class AgentController():
    def __init__(self):
        self.guard_agent = GuardAgent()
        self.classification_agent = ClassificationAgent()

        self.agent_dict: Dict[str, AgentProtocol] = {
            "CourseQA_Agent": QACourseAgent(),
            "EmailResponse_Agent": EmailResponseAgent(),
            "AssessmentGenerator_Agent": AssessmentGeneratorAgent(),
            "SubmissionEvaluator_Agent": SubmissionEvaluatorAgent(),
            "StudentInfo_Agent": StudentInfoAgent(),
            "Greeting_Agent": GreetingAgent()
        }
        
        # Store uploaded files for submission evaluator
        self.submission_files = {
            "rubric": None,
            "assignment_details": None,
            "submission": None
        }
        
    def handle_file_upload(self, files):
        """Handle file uploads and automatically assign them to submission evaluator categories."""
        if not files:
            return False
            
        # Clear previous files
        self.submission_files = {
            "rubric": None,
            "assignment_details": None,
            "submission": None
        }
        
        # Auto-assign files based on order or content analysis
        file_types = ["assignment_details", "rubric", "submission"]
        
        for i, file_data in enumerate(files):
            if i < len(file_types):
                file_type = file_types[i]
                
                # Create temporary file
                temp_dir = tempfile.mkdtemp()
                file_name = file_data.get('name', f'file_{i}')
                temp_path = os.path.join(temp_dir, f"{file_type}_{file_name}")
                
                # Handle file content
                try:
                    if 'content' in file_data:
                        # If file content is base64 encoded, decode it
                        try:
                            content = base64.b64decode(file_data['content'])
                            with open(temp_path, 'wb') as f:
                                f.write(content)
                        except Exception:
                            # If not base64, treat as text
                            with open(temp_path, 'w', encoding='utf-8') as f:
                                f.write(str(file_data['content']))
                    elif hasattr(file_data, 'read'):
                        # If it's a file-like object
                        with open(temp_path, 'wb') as f:
                            f.write(file_data.read())
                    else:
                        # Fallback: treat as string content
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            f.write(str(file_data))
                    
                    self.submission_files[file_type] = temp_path
                    
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    continue
                    
        return True
        
    def detect_submission_evaluation_intent(self, messages):
        """Detect if the user is asking for submission evaluation."""
        if not messages:
            return False
            
        last_message = messages[-1].get('content', '').lower()
        
        # Keywords that indicate submission evaluation
        evaluation_keywords = [
            'evaluate', 'evaluation', 'grade', 'grading', 'assess', 'assessment',
            'score', 'scoring', 'rubric', 'submission', 'assignment', 'feedback',
            'review', 'check my work', 'rate my', 'how did i do'
        ]
        
        return any(keyword in last_message for keyword in evaluation_keywords)
        
    def get_response(self, input):
        # Extract User Input
        job_input = input["input"]
        messages = job_input["messages"]
        files = job_input.get("files", [])
        
        # Handle file uploads if present
        files_uploaded = False
        if files:
            files_uploaded = self.handle_file_upload(files)
        
        # Get GuardAgent's response
        guard_agent_response = self.guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            return guard_agent_response
        
        # Check if files were uploaded or if user is asking for submission evaluation
        if files_uploaded or self.detect_submission_evaluation_intent(messages):
            # Force use of SubmissionEvaluator_Agent
            chosen_agent = "SubmissionEvaluator_Agent"
            
            # Check if all required files are uploaded
            missing_files = [file_type for file_type, path in self.submission_files.items() if path is None]
            
            if missing_files and not files_uploaded:
                # If files are missing and no new files uploaded, inform the user
                missing_files_str = ", ".join(missing_files)
                return {
                    "role": "assistant",
                    "content": f"To evaluate a submission, I need the following files: {missing_files_str}. Please upload them using the attachment button (📎) or drag and drop them into the chat.",
                    "memory": {}
                }
            elif missing_files and files_uploaded:
                # If some files are still missing after upload
                missing_files_str = ", ".join(missing_files)
                uploaded_files_str = ", ".join([file_type for file_type, path in self.submission_files.items() if path is not None])
                return {
                    "role": "assistant",
                    "content": f"I received: {uploaded_files_str}. I still need: {missing_files_str}. Please upload the remaining files to proceed with evaluation.",
                    "memory": {}
                }
            
            # If all files are available, proceed with evaluation
            agent = self.agent_dict[chosen_agent]
            
            try:
                # Load the files into the SubmissionEvaluatorAgent
                if self.submission_files["rubric"]:
                    agent.load_rubric(self.submission_files["rubric"], "text")
                if self.submission_files["assignment_details"]:
                    agent.load_assignment_details(self.submission_files["assignment_details"], "text")
                if self.submission_files["submission"]:
                    agent.load_submission(self.submission_files["submission"])
                
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
                
                # Clear the files after evaluation
                self.submission_files = {
                    "rubric": None,
                    "assignment_details": None,
                    "submission": None
                }
                
                return response
                
            except Exception as e:
                return {
                    "role": "assistant",
                    "content": f"Error during evaluation: {str(e)}. Please check your files and try again.",
                    "memory": {}
                }
        else:
            # For regular chat without files, use classification agent
            classification_agent_response = self.classification_agent.get_response(messages)
            chosen_agent = classification_agent_response["memory"]["classification_decision"]
            
            # Get the chosen agent's response
            agent = self.agent_dict.get(chosen_agent)
            if not agent:
                # Fallback to greeting agent
                agent = self.agent_dict["Greeting_Agent"]
            
            response = agent.get_response(messages)

        return response

