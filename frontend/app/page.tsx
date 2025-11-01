"use client"

import { useDATAmat } from "@/lib/datamat-context"
import { useDATAmatInit } from "@/hooks/use-datamat-init"
import { Navigation } from "@/components/navigation"
import { DashboardPage } from "@/components/dashboard-page"
import { ChatPage } from "@/components/chat-page"
import { UploadPage } from "@/components/upload-page"
import { KagglePage } from "@/components/kaggle-page"
import { DatasetsPage } from "@/components/datasets-page"

function PageContent() {
  const { currentPage } = useDATAmat()
  useDATAmatInit()

  return (
    <div className="min-h-screen bg-[#FAFAFA]">
      <Navigation />

      <main className="w-full px-4 md:px-6 py-4 md:py-8">
        <div className="max-w-7xl mx-auto">
          {currentPage === "dashboard" && <DashboardPage />}
          {currentPage === "chat" && <ChatPage />}
          {currentPage === "upload" && <UploadPage />}
          {currentPage === "kaggle" && <KagglePage />}
          {currentPage === "datasets" && <DatasetsPage />}
        </div>
      </main>
    </div>
  )
}

export default function HomePage() {
  return <PageContent />
}
