"use client"

import { useEffect } from "react"
import { useDATAmat } from "@/lib/datamat-context"

export function useDATAmatInit() {
  const { setIsHealthy, backendUrl, setDatasets, setError } = useDATAmat()

  useEffect(() => {
    const initializeApp = async () => {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000)

        const healthResponse = await fetch(`${backendUrl}/ai21/health`, {
          signal: controller.signal,
        })
        clearTimeout(timeoutId)

        if (!healthResponse.ok) {
          setIsHealthy(false)
          return
        }

        const healthData = await healthResponse.json()
        const isHealthy = healthData.status === "healthy"
        setIsHealthy(isHealthy)

        if (isHealthy) {
          try {
            const datasetsResponse = await fetch(`${backendUrl}/ai21/list-datasets`)
            if (datasetsResponse.ok) {
              const datasetsData = await datasetsResponse.json()
              setDatasets(datasetsData.datasets || [])
            }
          } catch (err) {
            console.error("Failed to load datasets:", err)
          }
        }
      } catch (err) {
        setIsHealthy(false)
        console.warn("Backend not available - app will work in offline mode")
      }
    }

    initializeApp()
  }, [backendUrl, setIsHealthy, setDatasets])
}
