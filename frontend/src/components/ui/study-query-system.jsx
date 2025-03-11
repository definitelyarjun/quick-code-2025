import React, { useState } from "react";
import { Search, CornerDownLeft, Upload, FileText } from "lucide-react";
import { Button } from "./button";
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "./chat-bubble";
import { ChatMessageList } from "./chat-message-list";
import { ChatInput } from "./chat-message-list";
import { FileUpload } from "./file-upload";
import { api } from "../../services/api";

export function StudyQuerySystem({ onQuery, isDarkMode }) {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [showUpload, setShowUpload] = useState(false);

  // Helper function to extract text from source objects
  const getSourceText = (source) => {
    // If source is a string, return it directly
    if (typeof source === 'string') return source;
    
    // If source has a text property, return that
    if (source.text) return source.text;
    
    // If source has a content property, return that
    if (source.content) return source.content;
    
    // If source is an object with chunk_index and score, it's from the backend
    if (source.chunk_index !== undefined) {
      // Convert the object to a string representation
      return `Chunk ${source.chunk_index} (Relevance: ${Math.round(source.score * 100)}%)`;
    }
    
    // Fallback: stringify the object but remove curly braces
    return JSON.stringify(source).replace(/[{}]/g, '');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Add user query to messages
    const userMessage = {
      type: 'user',
      content: query
    };
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);

    try {
      const response = await onQuery(query);
      
      // Process sources to ensure they're strings
      const processedSources = Array.isArray(response.sources) 
        ? response.sources.map(source => {
            return {
              original: source,
              text: getSourceText(source),
              page: source.page || source.metadata?.page || null
            };
          })
        : [];
      
      // Add AI response to messages
      const aiMessage = {
        type: 'ai',
        content: response.answer || "I couldn't find an answer to your question.",
        sources: processedSources
      };
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error querying:', error);
      // Add error message
      const errorMessage = {
        type: 'ai',
        content: "Sorry, there was an error processing your request.",
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setQuery(""); // Clear input after sending
    }
  };

  const handleFileUpload = async (file) => {
    try {
      const result = await api.uploadPDF(file);
      
      // Add system message about successful upload
      const systemMessage = {
        type: 'ai',
        content: `Successfully uploaded "${file.name}". You can now ask questions about this document.`,
        isSystem: true
      };
      
      setMessages(prev => [...prev, systemMessage]);
      setShowUpload(false);
    } catch (error) {
      console.error('Error uploading file:', error);
      
      // Add error message
      const errorMessage = {
        type: 'ai',
        content: `Failed to upload "${file.name}". Please try again.`,
        error: true,
        isSystem: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header with upload button */}
      <div className={`flex justify-between items-center mb-4 ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
        <h2 className="text-xl font-semibold">Study Assistant</h2>
        <Button
          onClick={() => setShowUpload(!showUpload)}
          className={`px-3 py-2 rounded-lg flex items-center gap-2 ${
            isDarkMode
              ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
          }`}
        >
          <Upload className="h-4 w-4" />
          {showUpload ? 'Hide Upload' : 'Upload PDF'}
        </Button>
      </div>

      {/* File upload area */}
      {showUpload && (
        <div className="mb-6">
          <FileUpload onUpload={handleFileUpload} isDarkMode={isDarkMode} />
        </div>
      )}

      <div className="flex-1 overflow-auto mb-4">
        <ChatMessageList>
          {messages.length === 0 && (
            <div className={`text-center py-8 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              <p className="text-lg mb-2">Ask me anything about your study materials</p>
              <p className="text-sm">I'll search through your documents and provide relevant answers</p>
            </div>
          )}

          {messages.map((message, index) => (
            <ChatBubble key={index} variant={message.type === 'user' ? 'sent' : 'received'}>
              <ChatBubbleMessage 
                variant={message.type === 'user' ? 'sent' : 'received'} 
                className={message.type === 'user' 
                  ? `${isDarkMode ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-900'}`
                  : `${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'} ${message.isSystem ? 'border-l-4 border-blue-500' : ''}`
                }
              >
                {message.type === 'user' ? (
                  message.content
                ) : (
                  <div className="space-y-4">
                    <p>{message.content}</p>
                    
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-4">
                        <p className={`text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          Sources:
                        </p>
                        <div className="space-y-2">
                          {message.sources.map((source, idx) => (
                            <div 
                              key={idx} 
                              className={`p-3 rounded-lg text-sm ${
                                isDarkMode 
                                  ? 'bg-gray-700 border border-gray-600' 
                                  : 'bg-gray-50 border border-gray-100'
                              }`}
                            >
                              <p className={isDarkMode ? 'text-gray-200' : 'text-gray-700'}>
                                {source.text}
                              </p>
                              {source.page && (
                                <p className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                  Page: {source.page}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </ChatBubbleMessage>
            </ChatBubble>
          ))}

          {isLoading && (
            <ChatBubble variant="received">
              <ChatBubbleMessage 
                isLoading 
                className={isDarkMode ? 'bg-gray-800' : 'bg-white'} 
              />
            </ChatBubble>
          )}
        </ChatMessageList>
      </div>

      <div className="mt-auto">
        <form onSubmit={handleSubmit}>
          <div className={`relative rounded-xl border focus-within:ring-2 ${
            isDarkMode 
              ? 'border-gray-700 bg-gray-800 focus-within:ring-gray-600 focus-within:border-gray-500' 
              : 'border-gray-200 bg-gray-50 focus-within:ring-gray-200 focus-within:border-gray-300'
          }`}>
            <ChatInput
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What do you want to know?"
              className={`min-h-[56px] w-full resize-none rounded-xl bg-transparent px-4 py-[1.3rem] text-base focus:outline-none ${
                isDarkMode ? 'text-white placeholder:text-gray-400' : 'text-gray-900 placeholder:text-gray-500'
              }`}
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center space-x-2">
              <Button 
                type="submit" 
                className={`rounded-lg px-4 py-2 flex items-center gap-2 transition-colors ${
                  isDarkMode
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                }`}
                disabled={!query.trim() || isLoading}
              >
                {isLoading ? "Thinking..." : "Ask"}
                <CornerDownLeft className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
