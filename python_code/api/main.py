from agent_controller import AgentController
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

agent_controller = AgentController()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("input", {})
    
    # The frontend sends messages and files in a specific format
    # The agent_controller expects 'messages' and 'files' within 'input'
    response = agent_controller.get_response({"input": user_input})
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
