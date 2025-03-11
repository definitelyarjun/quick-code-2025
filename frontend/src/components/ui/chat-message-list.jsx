import React, { useEffect, useRef } from "react";
import { cn } from "../../lib/utils";

export const ChatMessageList = ({ children, className, ...props }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [children]);

  return (
    <div
      className={cn("flex flex-col gap-4 overflow-y-auto p-4", className)}
      {...props}
    >
      {children}
      <div ref={messagesEndRef} />
    </div>
  );
};

export const ChatInput = React.forwardRef(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          "flex w-full rounded-md bg-transparent text-sm placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        rows={1}
        {...props}
      />
    );
  }
);

ChatInput.displayName = "ChatInput";
