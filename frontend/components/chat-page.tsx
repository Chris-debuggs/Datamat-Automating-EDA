"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { useDATAmat } from "@/lib/datamat-context"
import { Button } from "@/components/ui/button"
import { Send, Loader2 } from "lucide-react"

const EXAMPLE_QUESTIONS = [
  "What are the main columns in this dataset?",
  "Show me the data distribution",
  "What are the key statistics?",
  "Are there any missing values?",
]

export function ChatPage() {
  const { messages, addMessage, isLoading, setIsLoading, error, setError, backendUrl, datasets, activeDataset } =
    useDATAmat()
  const [input, setInput] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput("")
    addMessage(userMessage, true)
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${backendUrl}/ai21/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response")
      }

      const data = await response.json()
      addMessage(data.answer, false)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error connecting to backend")
      addMessage("Sorry, I encountered an error. Please try again.", false)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4 md:space-y-6">
      {datasets.length === 0 && (
        <div className="bg-[#3B4871] text-white p-4 md:p-6 border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] font-bold text-sm md:text-base">
          Upload a dataset first to ask questions about it.
        </div>
      )}

      <div className="bg-white border-[3px] border-[#3B4871] shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] p-4 md:p-6 min-h-[400px] md:min-h-[500px] flex flex-col">
        <div className="flex-1 overflow-y-auto space-y-3 md:space-y-4 mb-4 md:mb-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-center">
              <div>
                <p className="text-gray-600 font-bold text-base md:text-lg mb-4">
                  Start asking questions about your dataset
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 md:gap-3">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => {
                        setInput(q)
                      }}
                      className="p-2 md:p-3 text-xs md:text-sm font-bold border-[2px] border-[#3B4871] bg-white hover:bg-[#F8F9FC] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] transition-all text-left text-[#3B4871]"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] md:max-w-[70%] p-3 md:p-4 border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] font-bold text-xs md:text-sm ${
                    msg.isUser ? "bg-[#3B4871] text-white" : "bg-[#F8F9FC] text-[#3B4871] border-[#3B4871]"
                  }`}
                >
                  <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="p-3 md:p-4 border-[3px] border-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] bg-[#F8F9FC]">
                <Loader2 className="animate-spin text-[#3B4871]" size={20} />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex gap-2 flex-col sm:flex-row">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something about your data..."
            disabled={isLoading || datasets.length === 0}
            className="flex-1 p-2 md:p-3 border-[3px] border-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] bg-white font-bold text-xs md:text-sm focus:outline-none focus:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] focus:translate-x-[1px] focus:translate-y-[1px] transition-all disabled:opacity-50"
          />
          <Button
            type="submit"
            disabled={isLoading || !input.trim() || datasets.length === 0}
            className="bg-[#3B4871] text-white border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-bold px-4 py-2 disabled:opacity-50 h-auto"
          >
            <Send size={18} />
          </Button>
        </form>
      </div>
    </div>
  )
}
