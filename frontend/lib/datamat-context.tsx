"use client"

import type React from "react"
import { createContext, useContext, useState } from "react"

interface Message {
  text: string
  isUser: boolean
  timestamp: Date
}

interface Dataset {
  filename: string
  size_bytes: number
  created: string
}

interface DATAmatContextType {
  // Navigation
  currentPage: "dashboard" | "chat" | "upload" | "kaggle" | "datasets"
  setCurrentPage: (page: DATAmatContextType["currentPage"]) => void

  // Chat
  messages: Message[]
  addMessage: (text: string, isUser: boolean) => void
  clearMessages: () => void

  // Datasets
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  activeDataset: string | null
  setActiveDataset: (filename: string | null) => void

  // Loading states
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  error: string | null
  setError: (error: string | null) => void

  // Backend connection
  backendUrl: string
  isHealthy: boolean
  setIsHealthy: (healthy: boolean) => void
}

const DATAmatContext = createContext<DATAmatContextType | undefined>(undefined)

export function DATAmatProvider({ children }: { children: React.ReactNode }) {
  const [currentPage, setCurrentPage] = useState<DATAmatContextType["currentPage"]>("dashboard")
  const [messages, setMessages] = useState<Message[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [activeDataset, setActiveDataset] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isHealthy, setIsHealthy] = useState(false)

  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001"

  const addMessage = (text: string, isUser: boolean) => {
    setMessages((prev) => [...prev, { text, isUser, timestamp: new Date() }])
  }

  const clearMessages = () => {
    setMessages([])
  }

  return (
    <DATAmatContext.Provider
      value={{
        currentPage,
        setCurrentPage,
        messages,
        addMessage,
        clearMessages,
        datasets,
        setDatasets,
        activeDataset,
        setActiveDataset,
        isLoading,
        setIsLoading,
        error,
        setError,
        backendUrl,
        isHealthy,
        setIsHealthy,
      }}
    >
      {children}
    </DATAmatContext.Provider>
  )
}

export function useDATAmat() {
  const context = useContext(DATAmatContext)
  if (!context) {
    throw new Error("useDATAmat must be used within DATAmatProvider")
  }
  return context
}
