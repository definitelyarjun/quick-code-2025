import React, { useState } from "react";
import { Search, CornerDownLeft } from "lucide-react";
import { Button } from "./button";
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "./chat-bubble";
import { ChatMessageList } from "./chat-message-list";
import { ChatInput } from "./chat-message-list";

export function StudyQuerySystem({ onQuery }) {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);

    try {
      const response = await onQuery(query);
      setResults(response.relevant_chunks || []);
    } catch (error) {
      console.error('Error querying:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        <ChatMessageList>
          {hasSearched && (
            <ChatBubble variant="sent">
              <ChatBubbleMessage variant="sent" className="bg-gray-100 text-gray-900">
                {query}
              </ChatBubbleMessage>
            </ChatBubble>
          )}

          {isLoading && (
            <ChatBubble variant="received">
              <ChatBubbleMessage isLoading className="bg-white" />
            </ChatBubble>
          )}

          {results.length > 0 && (
            <ChatBubble variant="received">
              <ChatBubbleMessage variant="received" className="bg-white">
                <div className="space-y-4">
                  <p className="font-medium">Here are the most relevant passages from your study materials:</p>
                  {results.map((chunk, idx) => (
                    <div key={idx} className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                      <p className="text-gray-700">{chunk}</p>
                    </div>
                  ))}
                </div>
              </ChatBubbleMessage>
            </ChatBubble>
          )}

          {hasSearched && !isLoading && results.length === 0 && (
            <ChatBubble variant="received">
              <ChatBubbleMessage variant="received" className="bg-white">
                No relevant information found in your study materials. Try rephrasing your question.
              </ChatBubbleMessage>
            </ChatBubble>
          )}
        </ChatMessageList>
      </div>

      <div className="mt-4">
        <form onSubmit={handleSubmit}>
          <div className="relative rounded-xl border border-gray-200 bg-gray-50 focus-within:ring-2 focus-within:ring-gray-200 focus-within:border-gray-300">
            <ChatInput
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What do you want to know?"
              className="min-h-[56px] w-full resize-none rounded-xl bg-transparent px-4 py-[1.3rem] text-base focus:outline-none"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center space-x-2">
              <Button 
                type="submit" 
                className="bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg px-4 py-2 flex items-center gap-2 transition-colors"
                disabled={!query.trim() || isLoading}
              >
                {isLoading ? "Searching..." : "Search"}
                <CornerDownLeft className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
