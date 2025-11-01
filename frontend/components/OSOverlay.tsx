"use client"

import { useEffect } from "react"
import { useUIStore } from "@/lib/ui-store"
import { OrbSlot } from "./OrbSlot"
import { PersonaToggle } from "./PersonaToggle"
import { MiniAppAbout } from "./MiniAppAbout"
import { MiniAppResume } from "./MiniAppResume"
import { MiniAppWritings } from "./MiniAppWritings"
import { MiniAppArt } from "./MiniAppArt"
import { Button } from "@/components/ui/button"
import { X, User, FileText, PenTool, Palette } from "lucide-react"

const APP_COMPONENTS = {
  about: MiniAppAbout,
  resume: MiniAppResume,
  writings: MiniAppWritings,
  art: MiniAppArt,
}

const APP_ICONS = {
  about: User,
  resume: FileText,
  writings: PenTool,
  art: Palette,
}

export function OSOverlay() {
  const { osOpen, activeApp, closeOS, setActiveApp } = useUIStore()

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && osOpen) {
        closeOS()
      }
    }

    document.addEventListener("keydown", handleEscape)
    return () => document.removeEventListener("keydown", handleEscape)
  }, [osOpen, closeOS])

  if (!osOpen) return null

  const ActiveComponent = activeApp ? APP_COMPONENTS[activeApp] : null

  return (
    <div className="fixed inset-0 z-50 bg-[#FAFAFA] overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between p-4 border-b-[3px] border-black bg-white">
        <div className="flex items-center gap-4">
          <OrbSlot size="sm" />
          <h1 className="text-2xl font-black">MANAN OS</h1>
        </div>

        <div className="flex items-center gap-4">
          <PersonaToggle />
          <Button
            onClick={closeOS}
            className="w-10 h-10 p-0 bg-[#FF2E63] text-white border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all focus:ring-4 focus:ring-[#FF2E63]"
            aria-label="Close OS"
          >
            <X size={16} />
          </Button>
        </div>
      </header>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Sidebar */}
        <nav className="w-64 bg-white border-r-[3px] border-black p-4">
          <div className="space-y-2">
            {Object.entries(APP_COMPONENTS).map(([key, _]) => {
              const Icon = APP_ICONS[key as keyof typeof APP_ICONS]
              const isActive = activeApp === key

              return (
                <Button
                  key={key}
                  onClick={() => setActiveApp(key as any)}
                  className={`w-full justify-start gap-3 h-12 border-[3px] border-black font-bold text-left transition-all focus:ring-4 focus:ring-[#FF2E63] ${
                    isActive
                      ? "bg-[#FF2E63] text-white shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"
                      : "bg-white text-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px]"
                  }`}
                >
                  <Icon size={20} />
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </Button>
              )
            })}
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-auto">
          {ActiveComponent ? (
            <ActiveComponent />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <h2 className="text-4xl font-black mb-4">Welcome to Manan OS</h2>
                <p className="text-xl text-gray-600">Select an app from the sidebar to get started</p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
