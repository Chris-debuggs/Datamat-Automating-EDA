"use client"

import type React from "react"

import { useState } from "react"
import { useDATAmat } from "@/lib/datamat-context"
import { Button } from "@/components/ui/button"
import { Upload, FileUp, Check, AlertCircle } from "lucide-react"

const SUPPORTED_FORMATS = ["CSV", "PDF", "JSON", "TXT", "XLSX", "XLS"]

export function UploadPage() {
  const { setIsLoading, setError, backendUrl, setDatasets, error } = useDATAmat()
  const [file, setFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle")

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setUploadStatus("idle")
      setError(null)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const droppedFile = e.dataTransfer.files?.[0]
    if (droppedFile) {
      setFile(droppedFile)
      setUploadStatus("idle")
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${backendUrl}/ai21/upload-dataset`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Upload failed")
      }

      const data = await response.json()

      const listResponse = await fetch(`${backendUrl}/ai21/list-datasets`)
      const listData = await listResponse.json()
      setDatasets(listData.datasets)

      setUploadStatus("success")
      setFile(null)
      setTimeout(() => setUploadStatus("idle"), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed")
      setUploadStatus("error")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4 md:space-y-6 max-w-2xl">
      <div className="bg-white p-6 md:p-8 border-[3px] border-[#3B4871] shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <h2 className="text-2xl md:text-3xl font-black mb-6 text-[#3B4871]">Upload Dataset</h2>

        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="border-[3px] border-dashed border-[#3B4871] p-8 md:p-12 text-center bg-[#F8F9FC] cursor-pointer transition-all hover:bg-[#EFF0F7] mb-6"
        >
          <input
            type="file"
            onChange={handleFileSelect}
            accept={`.${SUPPORTED_FORMATS.join(",.")}`}
            className="hidden"
            id="file-input"
          />
          <label htmlFor="file-input" className="cursor-pointer flex flex-col items-center gap-3 md:gap-4">
            <FileUp size={40} className="text-[#3B4871] md:w-12 md:h-12" />
            <div>
              <p className="font-black text-base md:text-lg mb-2 text-[#3B4871]">Drag files here or click to select</p>
              <p className="text-xs md:text-sm text-[#3B4871] font-bold">Supported: {SUPPORTED_FORMATS.join(", ")}</p>
            </div>
          </label>
        </div>

        {file && (
          <div className="bg-[#F8F9FC] p-4 border-[3px] border-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] mb-6">
            <p className="font-bold mb-2 text-[#3B4871] text-sm md:text-base">Selected File:</p>
            <p className="text-xs md:text-sm font-bold text-[#3B4871] break-all">{file.name}</p>
            <p className="text-xs text-[#3B4871] font-bold mt-1">{(file.size / 1024).toFixed(2)} KB</p>
          </div>
        )}

        {uploadStatus === "success" && (
          <div className="flex gap-3 p-4 bg-green-50 border-[3px] border-green-500 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] mb-6 font-bold">
            <Check size={24} className="text-green-600 flex-shrink-0" />
            <div>
              <p className="font-black text-green-700 text-sm md:text-base">Upload successful!</p>
              <p className="text-xs md:text-sm text-green-600">Your dataset is ready for analysis.</p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex gap-3 p-4 bg-red-50 border-[3px] border-red-500 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] mb-6 font-bold">
            <AlertCircle size={24} className="text-red-600 flex-shrink-0" />
            <div>
              <p className="font-black text-red-700 text-sm md:text-base">Error</p>
              <p className="text-xs md:text-sm text-red-600">{error}</p>
            </div>
          </div>
        )}

        <Button
          onClick={handleUpload}
          disabled={!file}
          className="w-full bg-[#3B4871] text-white border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-black py-2 md:py-3 disabled:opacity-50 text-sm md:text-base"
        >
          <Upload size={20} className="mr-2" />
          Upload Dataset
        </Button>
      </div>

      <div className="bg-[#3B4871] text-white p-6 border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] font-bold">
        <p className="mb-2 md:mb-3 text-sm md:text-base">File Size Limit: 100 MB</p>
        <p className="text-xs md:text-sm">After upload, your dataset will be processed and ready for Q&A analysis.</p>
      </div>
    </div>
  )
}
