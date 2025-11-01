"use client"

import type React from "react"

import { useState } from "react"
import { useDATAmat } from "@/lib/datamat-context"
import { Button } from "@/components/ui/button"
import { Download, Check, AlertCircle, Loader2 } from "lucide-react"

export function KagglePage() {
  const { setIsLoading, setError, backendUrl, setDatasets, error, isLoading } = useDATAmat()
  const [datasetName, setDatasetName] = useState("")
  const [customFilename, setCustomFilename] = useState("")
  const [downloadStatus, setDownloadStatus] = useState<"idle" | "success" | "error">("idle")
  const [downloadedFiles, setDownloadedFiles] = useState<string[]>([])

  const handleDownload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!datasetName.trim()) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${backendUrl}/ai21/download-kaggle-dataset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_name: datasetName.trim(),
          ...(customFilename && { filename: customFilename }),
        }),
      })

      if (!response.ok) {
        throw new Error("Download failed")
      }

      const data = await response.json()
      setDownloadedFiles(data.files || [])

      const listResponse = await fetch(`${backendUrl}/ai21/list-datasets`)
      const listData = await listResponse.json()
      setDatasets(listData.datasets)

      setDownloadStatus("success")
      setDatasetName("")
      setCustomFilename("")
      setTimeout(() => setDownloadStatus("idle"), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed")
      setDownloadStatus("error")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4 md:space-y-6 max-w-2xl">
      <div className="bg-white p-6 md:p-8 border-[3px] border-[#3B4871] shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
        <h2 className="text-2xl md:text-3xl font-black mb-6 text-[#3B4871]">Download from Kaggle</h2>

        <form onSubmit={handleDownload} className="space-y-4 md:space-y-6">
          <div>
            <label className="block font-bold mb-2 text-[#3B4871] text-sm md:text-base">Dataset Name</label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="username/dataset-name"
              className="w-full p-2 md:p-3 border-[3px] border-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] bg-[#F8F9FC] font-bold text-xs md:text-sm focus:outline-none focus:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] focus:translate-x-[1px] focus:translate-y-[1px] transition-all text-gray-700"
            />
            <p className="text-xs text-[#3B4871] font-bold mt-2">
              Format: username/dataset-name (e.g., titanic or some-user/titanic-dataset)
            </p>
          </div>

          <div>
            <label className="block font-bold mb-2 text-[#3B4871] text-sm md:text-base">
              Custom Filename (Optional)
            </label>
            <input
              type="text"
              value={customFilename}
              onChange={(e) => setCustomFilename(e.target.value)}
              placeholder="Leave empty for default"
              className="w-full p-2 md:p-3 border-[3px] border-[#3B4871] shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] bg-[#F8F9FC] font-bold text-xs md:text-sm focus:outline-none focus:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] focus:translate-x-[1px] focus:translate-y-[1px] transition-all text-gray-700"
            />
          </div>

          {downloadStatus === "success" && (
            <div className="flex gap-3 p-4 bg-green-50 border-[3px] border-green-500 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] font-bold">
              <Check size={24} className="text-green-600 flex-shrink-0" />
              <div>
                <p className="font-black text-green-700 text-sm md:text-base">Download successful!</p>
                <p className="text-xs md:text-sm text-green-600">Downloaded files:</p>
                <ul className="text-xs md:text-sm mt-2 space-y-1 text-green-600">
                  {downloadedFiles.map((file) => (
                    <li key={file}>• {file}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {error && (
            <div className="flex gap-3 p-4 bg-red-50 border-[3px] border-red-500 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] font-bold">
              <AlertCircle size={24} className="text-red-600 flex-shrink-0" />
              <div>
                <p className="font-black text-red-700 text-sm md:text-base">Error</p>
                <p className="text-xs md:text-sm text-red-600">{error}</p>
              </div>
            </div>
          )}

          <Button
            type="submit"
            disabled={!datasetName.trim() || isLoading}
            className="w-full bg-[#3B4871] text-white border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-black py-2 md:py-3 disabled:opacity-50 flex justify-center items-center gap-2 text-sm md:text-base"
          >
            {isLoading ? (
              <>
                <Loader2 size={20} className="animate-spin" />
                Downloading...
              </>
            ) : (
              <>
                <Download size={20} />
                Download Dataset
              </>
            )}
          </Button>
        </form>
      </div>

      <div className="bg-[#3B4871] text-white p-6 border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] font-bold">
        <p className="mb-3 font-black text-sm md:text-base">How to Find Datasets</p>
        <p className="text-xs md:text-sm mb-3">
          Visit <span className="font-black">kaggle.com</span> and search for datasets. Once you find one, look at the
          URL or dataset page header to find the dataset name in the format:{" "}
          <span className="font-black">username/dataset</span>.
        </p>
        <p className="text-xs md:text-sm">Make sure you have Kaggle API credentials set up on the backend server.</p>
      </div>
    </div>
  )
}
