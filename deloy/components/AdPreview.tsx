"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Download, Share, Edit, ImageIcon } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import dynamic from "next/dynamic"

// Dynamically import fabric to avoid SSR issues
const FabricEditor = dynamic(() => import("@/components/FabricEditor"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[600px] flex items-center justify-center bg-gray-800">
      <div className="text-center text-white">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mb-2"></div>
        <p>Loading image editor...</p>
      </div>
    </div>
  ),
})

interface AdPreviewProps {
  generatedContent: any
  setGeneratedContent: (content: any) => void
}

export default function AdPreview({ generatedContent, setGeneratedContent }: AdPreviewProps) {
  const [activeTab, setActiveTab] = useState("preview")
  const [editedText, setEditedText] = useState("")
  const { toast } = useToast()

  useEffect(() => {
    if (generatedContent?.data?.results?.[0]?.ad_content) {
      setEditedText(generatedContent.data.results[0].ad_content)
    }
  }, [generatedContent])

  const copyToClipboard = () => {
    navigator.clipboard.writeText(editedText).then(() => {
      toast({
        title: "Copied to clipboard!",
        description: "Ad text has been copied to your clipboard.",
      })
    })
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="bg-gradient-to-r from-purple-900 to-blue-900 p-4 shadow-md">
        <Tabs defaultValue="preview" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2 bg-gray-800/50">
            <TabsTrigger
              value="preview"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-blue-500 data-[state=active]:text-white"
            >
              <ImageIcon size={16} />
              Preview
            </TabsTrigger>
            {generatedContent?.type !== "text" && (
              <TabsTrigger
                value="editor"
                className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-blue-500 data-[state=active]:text-white"
              >
                <Edit size={16} />
                Image Editor
              </TabsTrigger>
            )}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-auto">
        {activeTab === "preview" ? (
          <div className="p-6 max-w-4xl mx-auto">
            <Card className="bg-white shadow-xl transform-gpu hover:scale-[1.01] transition-transform duration-300">
              <CardContent className="p-6">
                <h2 className="text-2xl font-bold mb-4 text-purple-800">Generated Facebook Ad</h2>

                {generatedContent?.type !== "text" && (
                  <div className="mb-6 rounded-lg overflow-hidden shadow-lg">
                    {generatedContent.type === "product" && generatedContent.data.image_data && (
                      <img
                        src={`data:image/png;base64,${generatedContent.data.image_data}`}
                        alt="Generated Ad"
                        className="w-full h-auto"
                      />
                    )}
                    {generatedContent.type === "service" && generatedContent.data.image?.image_base64 && (
                      <img
                        src={`data:image/png;base64,${generatedContent.data.image.image_base64}`}
                        alt="Generated Ad"
                        className="w-full h-auto"
                      />
                    )}
                  </div>
                )}

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="adText" className="text-lg font-medium text-purple-800">
                      Ad Text
                    </Label>
                    <div className="relative">
                      <Textarea
                        id="adText"
                        value={editedText}
                        onChange={(e) => setEditedText(e.target.value)}
                        className="min-h-[200px] mt-2 hover:border-purple-500 focus:border-purple-500 transition-colors"
                      />
                      <Button variant="ghost" size="sm" className="absolute top-4 right-2" onClick={copyToClipboard}>
                        Copy
                      </Button>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      className="flex items-center gap-2 hover:bg-purple-100 transition-colors"
                      onClick={() => {
                        if (generatedContent?.type !== "text") {
                          setActiveTab("editor")
                        }
                      }}
                      disabled={generatedContent?.type === "text"}
                    >
                      <Edit size={16} />
                      Edit Image
                    </Button>

                    <Button
                      variant="outline"
                      className="flex items-center gap-2 hover:bg-purple-100 transition-colors"
                      disabled={generatedContent?.type === "text"}
                    >
                      <Download size={16} />
                      Download Image
                    </Button>

                    <Button
                      variant="outline"
                      className="flex items-center gap-2 hover:bg-purple-100 transition-colors"
                      onClick={copyToClipboard}
                    >
                      <Share size={16} />
                      Copy Text
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="p-6">
            {generatedContent?.type === "product" && generatedContent.data.image_data && (
              <FabricEditor imageData={generatedContent.data.image_data} initialText={editedText} />
            )}
            {generatedContent?.type === "service" && generatedContent.data.image?.image_base64 && (
              <FabricEditor imageData={generatedContent.data.image.image_base64} initialText={editedText} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
