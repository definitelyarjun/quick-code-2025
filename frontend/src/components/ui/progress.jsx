import React from "react";
import { cn } from "../../lib/utils";

export const Progress = React.forwardRef(({ className, value, max = 100, ...props }, ref) => {
  const percentage = value != null ? Math.min(Math.max(value, 0), max) : null;

  return (
    <div
      className={cn(
        "relative h-2 w-full overflow-hidden rounded-full bg-secondary",
        className
      )}
      {...props}
      ref={ref}
    >
      <div
        className="h-full w-full flex-1 bg-primary transition-all duration-500 ease-in-out"
        style={{
          transform: percentage != null ? `translateX(-${100 - percentage}%)` : "translateX(-100%)",
        }}
      />
    </div>
  );
});

Progress.displayName = "Progress";
