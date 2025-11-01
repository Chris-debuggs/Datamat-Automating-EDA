"use client"

import { createContext, useContext, type ReactNode, useState } from "react"

type Persona = "assistant" | "manan"
type AppType = "about" | "resume" | "writings" | "art" | null

interface UIContextType {
  persona: Persona
  osOpen: boolean
  activeApp: AppType
  setPersona: (persona: Persona) => void
  openOS: (app?: AppType) => void
  closeOS: () => void
  setActiveApp: (app: AppType) => void
}

const UIContext = createContext<UIContextType | undefined>(undefined)

export function UIProvider({ children }: { children: ReactNode }) {
  const [persona, setPersona] = useState<Persona>("assistant")
  const [osOpen, setOsOpen] = useState(false)
  const [activeApp, setActiveApp] = useState<AppType>(null)

  const openOS = (app: AppType = null) => {
    setOsOpen(true)
    setActiveApp(app)
  }

  const closeOS = () => {
    setOsOpen(false)
    setActiveApp(null)
  }

  return (
    <UIContext.Provider
      value={{
        persona,
        osOpen,
        activeApp,
        setPersona,
        openOS,
        closeOS,
        setActiveApp,
      }}
    >
      {children}
    </UIContext.Provider>
  )
}

export function useUIStore() {
  const context = useContext(UIContext)
  if (!context) {
    throw new Error("useUIStore must be used within UIProvider")
  }
  return context
}
