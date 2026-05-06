# -*- coding: utf-8 -*-
"""
This module defines the SubmissionEvaluatorAgent class.
"""
import os
import json
from pinecone import Pinecone
from openai import OpenAI
from copy import deepcopy
from .utils import get_chatbot_response, get_embedding, extract_text_from_pdf, read_text_file

class SubmissionEvaluatorAgent():
    """
    Agent responsible for evaluating student submissions based on assignment details and rubrics.
    Accepts assignment details, rubric, and submission as inputs in various formats.
    Uses Runpod LLM and Pinecone for context retrieval.
    """
    def __init__(self):
        """Initializes the SubmissionEvaluatorAgent with API clients and configuration."""
        # --- API Client Setup ---
        # Ensure environment variables are set before running
        runpod_token = os.getenv("RUNPOD_TOKEN")
        runpod_chat_url = os.getenv("RUNPOD_CHATBOT_URL")
        runpod_embedding_url = os.getenv("RUNPOD_EMBEDDING_URL")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME")
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct") # Default model

        if not all([runpod_token, runpod_chat_url, runpod_embedding_url, pinecone_api_key, pinecone_index]):
            raise ValueError("One or more required environment variables are missing.")

        self.client = OpenAI(
            api_key=runpod_token,
            base_url=runpod_chat_url,
        )
        self.embedding_client = OpenAI(
            api_key=runpod_token, 
            base_url=runpod_embedding_url
        )
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = pinecone_index
        self.namespace = "__default__" # Consider making this configurable

        # --- Data Storage ---
        self.assignment_details_text = None
        self.rubric_text = None
        self.submission_text = None
        self.course_context = None
        self.evaluation_results = None

        print("SubmissionEvaluatorAgent initialized.")

    def _load_input(self, input_data, input_type):
        """Helper function to load input based on type (pdf, text, message)."""
        if input_type == "pdf":
            print(f"Loading PDF: {input_data}")
            return extract_text_from_pdf(input_data)
        elif input_type == "text":
            print(f"Loading text file: {input_data}")
            return read_text_file(input_data)
        elif input_type == "message":
            print("Loading text from message.")
            return input_data # Direct text input
        else:
            print(f"Error: Unsupported input type 	{input_type}	 for data: {input_data}")
            return None

    def load_assignment_details(self, input_data, input_type):
        """Loads assignment details from PDF, text file, or direct message."""
        self.assignment_details_text = self._load_input(input_data, input_type)
        if self.assignment_details_text:
            print("Assignment details loaded successfully.")
        else:
            print("Failed to load assignment details.")

    def load_rubric(self, input_data, input_type):
        """Loads the rubric from PDF, text file, or direct message."""
        self.rubric_text = self._load_input(input_data, input_type)
        if self.rubric_text:
            print("Rubric loaded successfully.")
        else:
            print("Failed to load rubric.")

    def load_submission(self, pdf_path):
        """Loads the student submission from a PDF file."""
        print(f"Loading submission PDF: {pdf_path}")
        self.submission_text = extract_text_from_pdf(pdf_path)
        if self.submission_text:
            print("Submission loaded successfully.")
        else:
            print("Failed to load submission.")
            # Consider raising an error if submission loading fails, as it's critical
            # raise ValueError(f"Failed to extract text from submission PDF: {pdf_path}")

    def get_closest_results(self, input_embeddings, top_k=2):
        """Retrieves relevant documents from Pinecone based on embeddings."""
        try:
            index = self.pc.Index(self.index_name)
            results = index.query(
                vector=input_embeddings,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            print(f"Retrieved {len(results.get('matches', []))} results from Pinecone.")
            return results
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return None

    def retrieve_course_context(self, query_text, top_k=3):
        """Generates embedding for query and retrieves context from Pinecone."""
        print(f"Retrieving course context related to: \t{query_text[:100]}...\t")
        embedding_list = get_embedding(self.embedding_client, self.model_name, query_text)
        
        if not embedding_list:
            print("Error: Failed to generate embedding for context retrieval.")
            self.course_context = "Error: Could not retrieve relevant course context."
            return

        embedding = embedding_list[0]
        result = self.get_closest_results(embedding, top_k=top_k)

        if result and result.get("matches"):
            self.course_context = "\n".join([
                match["metadata"]["text"].strip() + "\n"
                for match in result["matches"]
            ])
            print("Course context retrieved successfully.")
        else:
            print("No relevant course context found or error during retrieval.")
            self.course_context = "No relevant course context found."
            
    def evaluate_submission(self):
        """Evaluates the submission based on details, rubric, and context using the LLM."""
        if not all([self.assignment_details_text, self.rubric_text, self.submission_text]):
            print("Error: Missing required data for evaluation (assignment details, rubric, or submission text).")
            self.evaluation_results = "Error: Cannot perform evaluation due to missing inputs."
            return

        # Retrieve context based on assignment details/rubric
        # Combine assignment details and rubric for a more focused context query
        context_query = f"Assignment: {self.assignment_details_text[:500]}\nRubric: {self.rubric_text[:500]}"
        self.retrieve_course_context(context_query)

        # Construct the prompt for the LLM
        system_prompt = """
You are an AI assistant specialized in evaluating student submissions. 
Your task is to provide a detailed evaluation based on the provided assignment details, rubric, student submission, and relevant course context. 
Structure your feedback clearly, addressing each rubric criterion. Provide specific examples from the submission to support your assessment. 
Identify strengths and weaknesses, and offer constructive suggestions for improvement. 
Output the evaluation in a structured format (e.g., Markdown).
"""

        user_prompt = f"""
Please evaluate the following student submission based on the provided assignment details, rubric, and course context.

**Assignment Details:**
{self.assignment_details_text}

**Rubric:**
{self.rubric_text}

**Relevant Course Context:**
{self.course_context}

**Student Submission Text:**
{self.submission_text}

**Evaluation Task:**
Provide a detailed evaluation addressing each rubric criterion, including scores (if applicable based on the rubric), strengths, weaknesses, and specific suggestions for improvement, citing evidence from the submission.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        print("Sending evaluation request to the LLM...")
        # Use the utility function to get the response
        evaluation_output = get_chatbot_response(self.client, self.model_name, messages)
        
        self.evaluation_results = evaluation_output
        print("Evaluation received from LLM.")
        # Further post-processing or structuring can be added here if needed

    def get_evaluation(self):
        """Returns the generated evaluation results."""
        return self.evaluation_results

