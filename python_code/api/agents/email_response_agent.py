# -*- coding: utf-8 -*-
"""
This module defines the EmailResponseAgent class.
"""
import os
from .utils import get_chatbot_response, get_embedding, double_check_json_output
from pinecone import Pinecone
from openai import OpenAI
from copy import deepcopy

class EmailResponseAgent():
    """
    Agent responsible for assisting with email drafting/responding.
    Accepts shared OpenAI client for efficiency.
    """
    def __init__(self):
        """Initializes the QACourseAgent with shared clients and config."""
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
        self.namespace = "__default__" + "Students_Instructors_Info"

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
        You are OrwaGPT, an AI assistant specialized in drafting and refining email communications for the **Smart System Management** course.
        Your role is to help students and instructors compose clear, professional, and effective emails related to course matters.

        
        USER REQUEST ANALYSIS:
        - Identify the purpose of the email (e.g., request extension, ask question, schedule meeting, provide information).
        - Determine the recipient (e.g., instructor, student, TA).
        - Extract key information from the user's request to include in the email.

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
            "memory": {"agent":"EmailResponse_Agent"
                      }
        }
        return output

