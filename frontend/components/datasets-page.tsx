"use client"

import { useEffect } from "react"
import { useDATAmat } from "@/lib/datamat-context"
import { Button } from "@/components/ui/button"
import { RefreshCw, FileIcon, Calendar, HardDrive } from "lucide-react"

export function DatasetsPage() {
  const { datasets, setDatasets, activeDataset, setActiveDataset, backendUrl, setIsLoading, setError } = useDATAmat()

  const loadDatasets = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${backendUrl}/ai21/list-datasets`)
      if (!response.ok) throw new Error("Failed to load datasets")
      const data = await response.json()
      setDatasets(data.datasets)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load datasets")
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadDatasets()
  }, [])

  const formatSize = (bytes: number) => {
    if (bytes === 0) return "0 B"
    const k = 1024
    const sizes = ["B", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  return (
    <div className="space-y-4 md:space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
        <h2 className="text-2xl md:text-3xl font-black text-[#3B4871]">Your Datasets</h2>
        <Button
          onClick={loadDatasets}
          className="bg-[#3B4871] text-white border-[3px] border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px] transition-all font-bold px-4 py-2 text-sm md:text-base h-auto"
        >
          <RefreshCw size={18} className="mr-2" />
          Refresh
        </Button>
      </div>

      {datasets.length === 0 ? (
        <div className="bg-white p-8 md:p-12 border-[3px] border-[#3B4871] shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-center">
          <p className="font-bold text-base md:text-lg text-[#3B4871] mb-4">No datasets uploaded yet</p>
          <p className="text-xs md:text-sm text-[#3B4871] font-bold">Upload a dataset to get started with analysis</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4">
          {datasets.map((dataset) => {
            return (
              <div
                key={dataset.filename}
                className={`p-4 md:p-6 border-[3px] border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] cursor-pointer transition-all ${
                  activeDataset === dataset.filename
                    ? "bg-[#3B4871] text-white"
                    : "bg-white border-[#3B4871] text-gray-700 hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[1px] hover:translate-y-[1px]"
                }`}
                onClick={() => setActiveDataset(dataset.filename)}
              >
                <div className="flex items-start gap-3 md:gap-4">
                  <FileIcon size={24} className="flex-shrink-0 mt-1 md:w-8 md:h-8" />
                  <div className="flex-1 min-w-0">
                    <h3 className="font-black text-sm md:text-lg break-all mb-3">{dataset.filename}</h3>
                    <div className="space-y-2 text-xs md:text-sm font-bold">
                      <div className="flex items-center gap-2">
                        <HardDrive size={14} className="md:w-4 md:h-4" />
                        <span>{formatSize(dataset.size_bytes)}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar size={14} className="md:w-4 md:h-4" />
                        <span>{new Date(dataset.created).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
