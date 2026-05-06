# -*- coding: utf-8 -*-
"""
This module defines the QACourseAgent class.
"""
import os
import json
from .utils import get_chatbot_response, get_embedding, double_check_json_output
from pinecone import Pinecone  # ✅ New SDK
from openai import OpenAI
from copy import deepcopy


class QACourseAgent():
    """
    Agent responsible for answering course-related questions using retrieved context.
    Accepts shared clients for efficiency.
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
        
        # ✅ Initialize Pinecone object
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.namespace = "__default__"  # Optional: make this configurable

    def get_closest_results(self, index_name, input_embeddings, top_k=2):
        # ✅ Use Pinecone.Index() through the pc instance
        index = self.pc.Index(index_name)
        
        results = index.query(
            vector=input_embeddings,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )
        return results

    def get_response(self, messages):
        messages = deepcopy(messages)

        user_message = messages[-1]['content']
        embedding = get_embedding(self.embedding_client, self.model_name, user_message)[0]

        result = self.get_closest_results(self.index_name, embedding)
        source_knowledge = "\n".join([
            x['metadata']['text'].strip() + '\n'
            for x in result['matches']
        ])

        prompt = f"""
        Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {user_message}
        """

        system_prompt = """
        You are an AI assistant for the **Smart System Management** course.
        Your primary role is to answer student and instructor questions accurately based on the provided course context.
        The course covers topics like risk management, system analysis, project management, IT infrastructure, data management, and cybersecurity within smart systems.
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        
        # # double check json 
        # chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
        
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self, output):
        return {
            "role": "assistant",
            "content": output,
            "memory": {"agent": "CourseQA_Agent"}
        }
