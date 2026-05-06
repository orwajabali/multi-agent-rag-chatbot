# -*- coding: utf-8 -*-
"""
This module defines the AssessmentGeneratorAgent class.
"""
import os
import json
from .utils import get_chatbot_response, get_embedding, double_check_json_output
from pinecone import Pinecone
from openai import OpenAI
from copy import deepcopy

class AssessmentGeneratorAgent():
    """
    Agent responsible for generating assessments.
    Accepts shared OpenAI client for efficiency.
    """
    def __init__(self):
        """Initializes the AssessmentGeneratorAgent with a shared API client and model name."""
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.embedding_client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"), 
            base_url=os.getenv("RUNPOD_EMBEDDING_URL")
        )
        self.model_name = os.getenv("MODEL_NAME")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        # Define namespace (consider making this configurable if needed)
        self.namespace = "__default__"

    def get_closest_results(self,index_name,input_embeddings,top_k=2):
        index = self.pc.Index(index_name)
        
        results = index.query(
            namespace=self.namespace,
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        return results

    def get_response(self,messages):
        messages = deepcopy(messages)

        user_message = messages[-1]['content']
        embedding = get_embedding(self.embedding_client,self.model_name,user_message)[0]
        result = self.get_closest_results(self.index_name,embedding)
        source_knowledge = "\n".join([x['metadata']['text'].strip()+'\n' for x in result['matches'] ])

        prompt = f"""
        Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {user_message}
        """

        system_prompt = f"""
        You are an AI assistant specialized in generating academic assessments for the **Smart System Management** course.
        Your task is to create high-quality assignments, quizzes, exam questions, practice exercises, or other assessment materials based on instructor requests.

        GUIDELINES:
        1.  **Course Focus**: All generated content MUST be relevant to the Smart System Management course content.
        2.  **Clarity & Specificity**: Ensure questions/tasks are clear, unambiguous, and appropriate for university-level students.
        3.  **Variety**: Generate diverse question types (e.g., multiple-choice, short answer, essay, problem-solving) if requested or appropriate.
        4.  **Context Awareness**: Use the provided conversation history or retrieved course context (if available) to tailor the assessment.
        5.  **Format**: Structure the output clearly. For quizzes/exams, number the questions. For assignments, provide clear instructions.

        USER REQUEST ANALYSIS:
        - Identify the type of assessment needed (quiz, assignment, exam questions, practice problems, course projects).
        - Determine the specific topic(s) to cover.
        - Note any constraints like the number of questions, difficulty level, or format.

        
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        
        # # double check json 
        # chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)

        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self,output):
        output = {
            "role": "assistant",
            "content": output,
            "memory": {"agent":"AssessmentGenerator_Agent"
                      }
        }
        return output

