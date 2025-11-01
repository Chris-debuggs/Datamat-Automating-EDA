interface OrbSlotProps {
  className?: string
  size?: "sm" | "md" | "lg"
}

export function OrbSlot({ className = "", size = "lg" }: OrbSlotProps) {
  const sizeClasses = {
    sm: "w-12 h-12",
    md: "w-24 h-24",
    lg: "w-32 h-32",
  }

  return (
    <div
      className={`${sizeClasses[size]} rounded-full bg-gradient-to-br from-[#FF2E63] to-[#FF6B9D] border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] ${className}`}
      role="img"
      aria-label="AI Assistant Orb"
    >
      {/* Slot for p5.js or SVG mascot */}
      <div className="w-full h-full rounded-full flex items-center justify-center text-white font-bold">AI</div>
    </div>
  )
}
