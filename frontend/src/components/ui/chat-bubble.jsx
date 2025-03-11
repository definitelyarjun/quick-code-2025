import React from "react";
import { cn } from "../../lib/utils";

export const ChatBubble = ({ children, variant = "sent", className, ...props }) => {
  return (
    <div
      className={cn(
        "flex w-full gap-2 p-4",
        variant === "received" && "justify-start",
        variant === "sent" && "justify-end",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

export const ChatBubbleAvatar = ({ src, fallback, className, ...props }) => {
  return (
    <div
      className={cn(
        "rounded-full bg-muted flex items-center justify-center text-xs font-medium",
        className
      )}
      {...props}
    >
      {src ? (
        <img src={src} alt="avatar" className="rounded-full" />
      ) : (
        <span>{fallback}</span>
      )}
    </div>
  );
};

export const ChatBubbleMessage = ({
  children,
  variant = "sent",
  isLoading,
  className,
  ...props
}) => {
  return (
    <div
      className={cn(
        "rounded-lg px-4 py-2 max-w-[80%]",
        variant === "received" && "bg-muted",
        variant === "sent" && "bg-primary text-primary-foreground",
        className
      )}
      {...props}
    >
      {isLoading ? (
        <div className="flex items-center gap-2">
          <div className="size-2 bg-current rounded-full animate-bounce" />
          <div className="size-2 bg-current rounded-full animate-bounce [animation-delay:-.3s]" />
          <div className="size-2 bg-current rounded-full animate-bounce [animation-delay:-.5s]" />
        </div>
      ) : (
        children
      )}
    </div>
  );
};
