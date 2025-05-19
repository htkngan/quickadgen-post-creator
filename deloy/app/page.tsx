"use client"

import { useState, useEffect } from "react"
import { Toaster } from "@/components/ui/toaster"
import { useToast } from "@/components/ui/use-toast"
import LoadingSpinner from "@/components/LoadingSpinner"
import AdForm from "@/components/AdForm"
import AdPreview from "@/components/AdPreview"
import ApiKeyModal from "@/components/ApiKeyModal"
import { Button } from "@/components/ui/button"
import { Settings } from "lucide-react"

export default function Home() {
  const [generatedContent, setGeneratedContent] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("generator")
  const [apiKey, setApiKey] = useState("")
  const [apiUrl, setApiUrl] = useState("http://localhost:8000")
  const [showApiModal, setShowApiModal] = useState(false)
  const { toast } = useToast()

  // Check for saved API key on component mount
  useEffect(() => {
    const savedApiKey = localStorage.getItem("fb_ad_api_key")
    const savedApiUrl = localStorage.getItem("fb_ad_api_url")

    if (savedApiKey) {
      setApiKey(savedApiKey)
    }

    if (savedApiUrl) {
      setApiUrl(savedApiUrl)
    }
  }, [])

  const handleSaveApiKey = (key: string, url: string) => {
    setApiKey(key)
    setApiUrl(url)
    localStorage.setItem("fb_ad_api_key", key)
    localStorage.setItem("fb_ad_api_url", url)
    setShowApiModal(false)

    toast({
      title: "API configuration saved",
      description: "Your API settings have been saved successfully.",
    })
  }

  const handleGeneratedContent = (content: any) => {
    setGeneratedContent(content)
    setActiveTab("preview")
    toast({
      title: "Ad generated successfully!",
      description: "Your ad has been created. You can now edit and customize it.",
    })
  }

  return (
    <main className="flex min-h-screen flex-col perspective-1000">
      <div className="flex justify-between items-center p-4 bg-gradient-to-r from-purple-600 to-blue-500 z-10">
        <h1 className="text-2xl font-bold text-white">Facebook Ad Generator</h1>
        <div className="flex space-x-2">
          <Button
            onClick={() => setActiveTab("generator")}
            className={`px-4 py-2 rounded-md ${
              activeTab === "generator" ? "bg-white text-purple-600" : "bg-purple-700 text-white"
            }`}
          >
            Generator
          </Button>
          <Button
            onClick={() => setActiveTab("preview")}
            className={`px-4 py-2 rounded-md ${
              activeTab === "preview" ? "bg-white text-purple-600" : "bg-purple-700 text-white"
            } ${!generatedContent ? "opacity-50 cursor-not-allowed" : ""}`}
            disabled={!generatedContent}
          >
            Preview & Edit
          </Button>
          <Button
            variant="outline"
            size="icon"
            className="bg-purple-700 text-white hover:bg-purple-800"
            onClick={() => setShowApiModal(true)}
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {isLoading && <LoadingSpinner />}

      <div className="flex-1 relative">
        {activeTab === "generator" ? (
          <div className="w-full h-[calc(100vh-64px)] bg-gradient-to-b from-purple-900 to-blue-900 flex items-center justify-center overflow-hidden">
            {/* 3D Background Elements */}
            <div className="scene-3d">
              <div className="floating-cube cube-1"></div>
              <div className="floating-cube cube-2"></div>
              <div className="floating-cube cube-3"></div>
              <div className="floating-cube cube-4"></div>
              <div className="floating-cube cube-5"></div>

              <div className="glow-sphere sphere-1"></div>
              <div className="glow-sphere sphere-2"></div>
              <div className="glow-sphere sphere-3"></div>

              <div className="grid-floor"></div>
            </div>

            <div className="absolute inset-0 overflow-hidden">
              <div className="absolute top-10 left-10 w-32 h-32 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
              <div className="absolute top-40 right-10 w-32 h-32 bg-yellow-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
              <div className="absolute -bottom-8 left-20 w-32 h-32 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
              <div className="absolute bottom-40 right-20 w-32 h-32 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
            </div>

            <div className="relative z-10 transform-gpu hover:scale-105 transition-transform duration-300">
              <AdForm
                onGenerateContent={handleGeneratedContent}
                setIsLoading={setIsLoading}
                apiKey={apiKey}
                apiUrl={apiUrl}
              />
            </div>
          </div>
        ) : (
          <div className="w-full h-[calc(100vh-64px)] bg-gradient-to-b from-gray-900 to-blue-900">
            {generatedContent && (
              <AdPreview generatedContent={generatedContent} setGeneratedContent={setGeneratedContent} />
            )}
          </div>
        )}
      </div>

      <ApiKeyModal
        isOpen={showApiModal}
        onClose={() => setShowApiModal(false)}
        onSave={handleSaveApiKey}
        initialApiKey={apiKey}
        initialApiUrl={apiUrl}
      />

      <Toaster />
    </main>
  )
}
