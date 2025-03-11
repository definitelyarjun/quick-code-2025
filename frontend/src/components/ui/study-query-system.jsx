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
    <div className="h-[600px] border bg-background rounded-lg flex flex-col">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold">Study Material Query System</h2>
        <p className="text-sm text-muted-foreground">
          Ask questions about your study materials to find relevant information
        </p>
      </div>
      
      <div className="flex-1 overflow-hidden">
        <ChatMessageList>
          {hasSearched && (
            <ChatBubble variant="sent">
              <ChatBubbleAvatar
                className="h-8 w-8 shrink-0"
                fallback="U"
              />
              <ChatBubbleMessage variant="sent">
                {query}
              </ChatBubbleMessage>
            </ChatBubble>
          )}

          {isLoading && (
            <ChatBubble variant="received">
              <ChatBubbleAvatar
                className="h-8 w-8 shrink-0"
                fallback="AI"
              />
              <ChatBubbleMessage isLoading />
            </ChatBubble>
          )}

          {results.length > 0 && (
            <ChatBubble variant="received">
              <ChatBubbleAvatar
                className="h-8 w-8 shrink-0"
                fallback="AI"
              />
              <ChatBubbleMessage variant="received">
                <div className="space-y-4">
                  <p className="font-medium">Here are the most relevant passages from your study materials:</p>
                  {results.map((chunk, idx) => (
                    <div key={idx} className="p-3 bg-muted/50 rounded-md">
                      <p className="text-sm">{chunk}</p>
                    </div>
                  ))}
                </div>
              </ChatBubbleMessage>
            </ChatBubble>
          )}

          {hasSearched && !isLoading && results.length === 0 && (
            <ChatBubble variant="received">
              <ChatBubbleAvatar
                className="h-8 w-8 shrink-0"
                fallback="AI"
              />
              <ChatBubbleMessage variant="received">
                No relevant information found in your study materials. Try rephrasing your question.
              </ChatBubbleMessage>
            </ChatBubble>
          )}
        </ChatMessageList>
      </div>

      <div className="p-4 border-t">
        <form
          onSubmit={handleSubmit}
          className="relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-1"
        >
          <ChatInput
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your study materials..."
            className="min-h-12 resize-none rounded-lg bg-background border-0 p-3 shadow-none focus-visible:ring-0"
          />
          <div className="flex items-center p-3 pt-0 justify-between">
            <div className="flex">
              <Button
                variant="ghost"
                size="icon"
                type="button"
                disabled={isLoading}
              >
                <Search className="h-4 w-4" />
              </Button>
            </div>
            <Button 
              type="submit" 
              size="sm" 
              className="ml-auto gap-1.5"
              disabled={!query.trim() || isLoading}
            >
              {isLoading ? "Searching..." : "Search"}
              <CornerDownLeft className="h-3.5 w-3.5" />
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
