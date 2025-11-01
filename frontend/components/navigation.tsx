"use client"

import { useState } from "react"
import { useDATAmat } from "@/lib/datamat-context"
import { Button } from "@/components/ui/button"
import { BarChart3, MessageSquare, Upload, Download, Database, Menu, X } from "lucide-react"

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "chat", label: "Ask Questions", icon: MessageSquare },
  { id: "upload", label: "Upload Data", icon: Upload },
  { id: "kaggle", label: "Kaggle", icon: Download },
  { id: "datasets", label: "Datasets", icon: Database },
]

export function Navigation() {
  const { currentPage, setCurrentPage, isHealthy } = useDATAmat()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const handleNavClick = (id: string) => {
    setCurrentPage(id as any)
    setMobileMenuOpen(false)
  }

  return (
    <nav className="flex items-center justify-between p-3 md:p-4 bg-white border-b-[3px] border-black shadow-[0_4px_0px_0px_rgba(0,0,0,1)]">
      {/* Logo */}
      <h1 className="text-2xl md:text-3xl font-black tracking-tight text-[#3B4871]">DATAmat</h1>

      {/* Desktop Navigation */}
      <div className="hidden md:flex gap-2">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <Button
            key={id}
            onClick={() => handleNavClick(id)}
            className={`flex items-center gap-2 px-4 py-2 h-10 font-bold border-[2px] border-black transition-all ${
              currentPage === id
                ? "bg-[#3B4871] text-white shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"
                : "bg-white text-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px]"
            }`}
          >
            <Icon size={18} />
            <span>{label}</span>
          </Button>
        ))}
      </div>

      {/* Status & Hamburger */}
      <div className="flex items-center gap-3">
        {/* Status Indicator */}
        <div
          className="flex items-center gap-1 md:gap-2 px-2 md:px-4 py-2 border-[2px] border-black bg-white font-bold"
          title={isHealthy ? "Backend is connected" : "Backend is not available - offline mode"}
        >
          <div className={`w-3 h-3 rounded-full ${isHealthy ? "bg-green-500" : "bg-red-500"}`} />
          <span className="text-xs md:text-sm hidden sm:inline text-[#3B4871]">
            {isHealthy ? "Connected" : "Offline"}
          </span>
        </div>

        {/* Hamburger Menu - Mobile */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 border-[2px] border-black bg-white hover:bg-[#F8F9FC] transition-all"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <X size={24} className="text-[#3B4871]" /> : <Menu size={24} className="text-[#3B4871]" />}
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="absolute top-[70px] left-0 right-0 bg-white border-b-[3px] border-black shadow-[0_4px_0px_0px_rgba(0,0,0,1)] md:hidden z-50">
          <div className="flex flex-col p-3 gap-2">
            {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => handleNavClick(id)}
                className={`flex items-center gap-3 px-4 py-3 w-full font-bold border-[2px] border-black transition-all ${
                  currentPage === id
                    ? "bg-[#3B4871] text-white shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"
                    : "bg-white text-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] active:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] active:translate-x-[1px] active:translate-y-[1px]"
                }`}
              >
                <Icon size={20} />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </nav>
  )
}
