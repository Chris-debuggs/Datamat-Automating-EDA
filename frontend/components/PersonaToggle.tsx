"use client"

import { useUIStore } from "@/lib/ui-store"
import { Button } from "@/components/ui/button"

export function PersonaToggle() {
  const { persona, setPersona } = useUIStore()

  return (
    <Button
      onClick={() => setPersona(persona === "assistant" ? "manan" : "assistant")}
      className="bg-white text-black border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-bold px-4 py-2 focus:ring-4 focus:ring-[#FF2E63]"
    >
      {persona === "assistant" ? "🤖 Assistant" : "👨‍💻 Manan"}
    </Button>
  )
}
