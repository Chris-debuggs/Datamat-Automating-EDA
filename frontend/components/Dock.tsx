"use client"

import { useUIStore } from "@/lib/ui-store"
import { Button } from "@/components/ui/button"
import { User, FileText, PenTool, Palette } from "lucide-react"

const DOCK_ITEMS = [
  { id: "about", label: "About", icon: User },
  { id: "resume", label: "Resume", icon: FileText },
  { id: "writings", label: "Writings", icon: PenTool },
  { id: "art", label: "Art", icon: Palette },
]

export function Dock() {
  const { openOS } = useUIStore()

  return (
    <div className="flex gap-3 p-4 bg-white border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
      {DOCK_ITEMS.map(({ id, label, icon: Icon }) => (
        <Button
          key={id}
          onClick={() => openOS(id as any)}
          className="w-12 h-12 p-0 bg-[#FF2E63] text-white border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all"
          aria-label={label}
        >
          <Icon size={20} />
        </Button>
      ))}
    </div>
  )
}
