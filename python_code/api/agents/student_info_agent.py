# -*- coding: utf-8 -*-
"""
This module defines the StudentInfoAgent class.
"""
import os
import json
from .utils import get_chatbot_response, get_embedding, double_check_json_output
from pinecone import Pinecone
from openai import OpenAI
from copy import deepcopy

class StudentInfoAgent():
    """
    Agent responsible for retrieving and providing information about enrolled students.
    Accepts shared clients for efficiency.
    """
    def __init__(self):
        """Initializes the StudentInfoAgent with shared clients and config."""
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
        self.namespace = "Students_Instructors_Info" 

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
        You are an AI assistant for the **Smart System Management** course instructor.
        The instructo of Smart System Management is Professor Allam Mousa
        Your task is to answer questions about students enrolled in the course using the provided student data context.
        You have access to information like student names, registration numbers, emails, region, descriptions, image paths, year, department, and status.
              
        GUIDELINES:
        1.  **Answer Based on Context**: ONLY use the provided CONTEXT to answer the user's query. Do not make up information.
        2.  **Specificity**: If the query asks for specific details (e.g., email adresse of a student), provide only that detail if found.
        3.  **Summarization**: If the query is general (e.g., "Tell me about Walid AbuZainah"), summarize the available information from the context.
        4.  **List Format**: If asked for a list (e.g., "List all active students"), present the information clearly.
        5.  **Not Found**: If the requested information is not in the CONTEXT, state that clearly (e.g., "I could not find information matching your query in the student database.").
        6.  **Privacy**: Do not infer or provide information beyond what's explicitly in the context.


      
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
            "memory": {"agent":"StudentInfo_Agent"
                      }
        }
        return output

