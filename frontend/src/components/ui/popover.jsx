import React, { useRef, useEffect } from "react";
import { cn } from "../../lib/utils";

const PopoverContext = React.createContext({});

export const Popover = ({ children }) => {
  const [open, setOpen] = React.useState(false);
  return (
    <PopoverContext.Provider value={{ open, setOpen }}>
      {children}
    </PopoverContext.Provider>
  );
};

export const PopoverTrigger = React.forwardRef(({ children, asChild = false, ...props }, ref) => {
  const { setOpen } = React.useContext(PopoverContext);
  const Comp = asChild ? React.cloneElement(children) : "button";

  if (asChild) {
    return React.cloneElement(children, {
      ref,
      onClick: (e) => {
        children.props.onClick?.(e);
        setOpen(v => !v);
      },
      ...props
    });
  }

  return (
    <Comp
      ref={ref}
      onClick={() => setOpen(v => !v)}
      {...props}
    >
      {children}
    </Comp>
  );
});
PopoverTrigger.displayName = "PopoverTrigger";

export const PopoverContent = React.forwardRef(({ 
  className,
  align = "center",
  sideOffset = 4,
  children,
  ...props 
}, ref) => {
  const { open, setOpen } = React.useContext(PopoverContext);
  const popoverRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target)) {
        setOpen(false);
      }
    };

    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open, setOpen]);

  if (!open) return null;

  return (
    <div
      ref={popoverRef}
      className={cn(
        "absolute z-50 w-72 rounded-md border bg-popover p-4 text-popover-foreground shadow-md outline-none",
        "animate-in fade-in-0 zoom-in-95",
        {
          "left-0": align === "start",
          "right-0": align === "end",
          "left-1/2 -translate-x-1/2": align === "center",
        },
        className
      )}
      style={{
        marginTop: sideOffset
      }}
      {...props}
    >
      {children}
    </div>
  );
});
PopoverContent.displayName = "PopoverContent";
