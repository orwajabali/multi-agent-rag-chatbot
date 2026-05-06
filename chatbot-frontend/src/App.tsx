import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  files?: File[];
}

function App() {
  const [input, setInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const API_URL = 'http://localhost:8080/chat';

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleFileUpload = (files: FileList | null) => {
    if (!files) return;
    
    const fileArray = Array.from(files);
    setUploadedFiles(prev => [...prev, ...fileArray]);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFileUpload(e.dataTransfer.files);
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() && uploadedFiles.length === 0) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      files: uploadedFiles.length > 0 ? [...uploadedFiles] : undefined
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setUploadedFiles([]);
    setIsLoading(true);

    try {
      const requestPayload = {
        input: {
          messages: [
            ...messages.map(msg => ({ role: msg.role, content: msg.content })),
            { role: 'user', content: input }
          ]
        }
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      });

      const data = await response.json();

      if (data.content) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.content }]);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGetStarted = () => {
    // Scroll to chat section when Get Started is clicked
    document.getElementById('chat-section')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="app">
      {/* Welcome Screen */}
      <div className="welcome-screen">
        <div className="welcome-content">
          <h1>Welcome to An-Najah INNOPARK Academy</h1>
          <div className="icon-container">
            <div className="INNOPARK-icon"></div>
            <div className="NAJAH-icon"></div>
          </div>
          <p className="tagline">"Your AI-powered learning companion"</p>
          <p className="subtitle">"Courses You Can Talk To"</p>
          
          <div className="robot-icon">
            <img src="/SSM_chatbot1.png" alt="" />
          </div>
          
          <h2>Smart System Management Chatbot</h2>
          
          <button className="get-started-btn" onClick={handleGetStarted}>
            Get Started
          </button>
        </div>
        
        <div className="bottom-indicator"></div>
      </div>
      
      {/* Chat Section */}
      <div id="chat-section" className="chat-container">
        <div className="chat-header">
          <h2>An-Najah INNOPARK Academy</h2>
        </div>
        
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <p>Start a conversation with the OrwaGPT, What can I help with?</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div 
                key={index} 
                className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
              >
                <div className="message-content">
                  {msg.content}
                  {msg.files && msg.files.length > 0 && (
                    <div className="message-files">
                      {msg.files.map((file, fileIndex) => (
                        <div key={fileIndex} className="file-indicator">
                          📎 {file.name}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message assistant-message">
              <div className="message-content loading">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* File Upload Area */}
        {uploadedFiles.length > 0 && (
          <div className="uploaded-files-section">
            <div className="uploaded-files">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="uploaded-file">
                  <span>📎 {file.name}</span>
                  <button 
                    type="button"
                    onClick={() => removeFile(index)}
                    className="remove-file-btn"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Drag and Drop Overlay */}
        {isDragOver && (
          <div className="drag-overlay">
            <div className="drag-content">
              <div className="drag-icon">📁</div>
              <p>Drop files here to upload</p>
            </div>
          </div>
        )}
        
        <form 
          className="input-form" 
          onSubmit={handleSubmit}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask OrwaGPT anything about Smart System Management Course..."
              disabled={isLoading}
            />
            <button 
              type="button"
              className="attachment-btn"
              onClick={() => fileInputRef.current?.click()}
              title="Attach files"
            >
              📎
            </button>
          </div>
          <button type="submit" disabled={isLoading || (!input.trim() && uploadedFiles.length === 0)}>
            Send
          </button>
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => handleFileUpload(e.target.files)}
            style={{ display: 'none' }}
            accept=".pdf,.doc,.docx,.txt"
          />
        </form>
      </div>
    </div>
  );
}

export default App;

