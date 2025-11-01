"use client"

import { useDATAmat } from "@/lib/datamat-context"
import { Button } from "@/components/ui/button"
import { MessageSquare, Upload, AlertCircle } from "lucide-react"

export function DashboardPage() {
  const { datasets, setCurrentPage, isHealthy } = useDATAmat()

  return (
    <div className="space-y-6 md:space-y-8">
      {!isHealthy && (
        <div className="flex gap-3 p-4 bg-red-500 text-white border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] font-bold">
          <AlertCircle size={24} className="flex-shrink-0" />
          <div>
            <p className="font-black text-sm md:text-base">Backend Not Connected</p>
            <p className="text-xs md:text-sm mt-1">
              Make sure your FastAPI server is running on http://localhost:8001 (or set NEXT_PUBLIC_BACKEND_URL
              environment variable)
            </p>
          </div>
        </div>
      )}

      <div className="bg-[#3B4871] text-white p-8 border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <h2 className="text-4xl font-black mb-4">Welcome to DATAmat</h2>
        <p className="text-lg font-bold mb-6">
          AI-powered exploratory data analysis. Upload your datasets and ask questions about them.
        </p>
        <div className="flex gap-4 flex-wrap">
          <Button
            onClick={() => setCurrentPage("upload")}
            disabled={!isHealthy}
            className="bg-white text-[#3B4871] border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-bold px-6 py-3 disabled:opacity-50"
          >
            <Upload size={20} className="mr-2" />
            Upload Dataset
          </Button>
          <Button
            onClick={() => setCurrentPage("chat")}
            disabled={datasets.length === 0}
            className="bg-white text-[#3B4871] border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-bold px-6 py-3 disabled:opacity-50"
          >
            <MessageSquare size={20} className="mr-2" />
            Ask Questions
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-6 border-[3px] border-[#3B4871] shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-sm font-bold text-[#3B4871] mb-2">Total Datasets</h3>
          <p className="text-4xl font-black text-[#3B4871]">{datasets.length}</p>
        </div>
        <div className="bg-white p-6 border-[3px] border-[#3B4871] shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-sm font-bold text-[#3B4871] mb-2">Last Upload</h3>
          <p className="text-lg font-bold text-[#3B4871]">
            {datasets.length > 0 ? new Date(datasets[datasets.length - 1].created).toLocaleDateString() : "None yet"}
          </p>
        </div>
        <div className="bg-white p-6 border-[3px] border-[#3B4871] shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-sm font-bold text-[#3B4871] mb-2">Total Size</h3>
          <p className="text-lg font-bold text-[#3B4871]">
            {(datasets.reduce((sum, d) => sum + d.size_bytes, 0) / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
      </div>

      <div className="bg-[#F8F9FC] p-8 border-[3px] border-[#3B4871] shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <h3 className="text-2xl font-black mb-4 text-[#3B4871]">Quick Start</h3>
        <ol className="space-y-3 font-bold">
          <li className="flex gap-3">
            <span className="bg-[#3B4871] text-white w-8 h-8 flex items-center justify-center border-[2px] border-black font-black">
              1
            </span>
            <span className="text-gray-700">Upload a CSV, JSON, PDF, or Excel file</span>
          </li>
          <li className="flex gap-3">
            <span className="bg-[#3B4871] text-white w-8 h-8 flex items-center justify-center border-[2px] border-black font-black">
              2
            </span>
            <span className="text-gray-700">Go to Ask Questions and start exploring</span>
          </li>
          <li className="flex gap-3">
            <span className="bg-[#3B4871] text-white w-8 h-8 flex items-center justify-center border-[2px] border-black font-black">
              3
            </span>
            <span className="text-gray-700">Get AI-powered insights about your data</span>
          </li>
        </ol>
      </div>
    </div>
  )
}
