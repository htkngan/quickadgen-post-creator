"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Download, Share, ImageIcon } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { generateAd, generateProductAd, generateImageService } from "@/lib/api"

interface AdPreviewProps {
  generatedContent: any
  setGeneratedContent: (content: any) => void
  apiKey: string
  apiUrl: string
  setIsLoading?: (loading: boolean) => void
}

export default function AdPreview({ generatedContent, setGeneratedContent, apiKey, apiUrl, setIsLoading }: AdPreviewProps) {
  const [activeTab, setActiveTab] = useState("preview")
  const [editedText, setEditedText] = useState("")
  const [showNoText, setShowNoText] = useState(false)
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

  // Regenerate content only
  const handleRegenerateContent = async () => {
    if (!generatedContent?.formData || !generatedContent?.type) return
    setIsLoading?.(true)
    try {
      const data = new FormData()
      Object.entries(generatedContent.formData).forEach(([key, value]) => {
        data.append(key, value as string)
      })
      let result
      if (generatedContent.type === "product") {
        result = await generateAd(data, apiUrl, apiKey)
        setGeneratedContent({ ...generatedContent, data: { ...generatedContent.data, results: result.results, image_data: generatedContent.data.image_data } })
      } else if (generatedContent.type === "service") {
        result = await generateAd(data, apiUrl, apiKey)
        setGeneratedContent({ ...generatedContent, data: { ...generatedContent.data, results: result.results, image: generatedContent.data.image } })
      } else {
        return
      }
      toast({ title: "Content regenerated!", description: "Ad content has been regenerated." })
    } catch (e) {
      toast({ title: "Error", description: "Failed to regenerate content.", variant: "destructive" })
    } finally {
      setIsLoading?.(false)
    }
  }

  // Regenerate image only
  const handleRegenerateImage = async () => {
    if (!generatedContent?.formData || !generatedContent?.type) return
    setIsLoading?.(true)
    try {
      const data = new FormData()
      Object.entries(generatedContent.formData).forEach(([key, value]) => {
        data.append(key, value as string)
      })
      let result
      if (generatedContent.type === "product") {
        result = await generateProductAd(data, apiUrl, apiKey)
        setGeneratedContent({ ...generatedContent, data: { ...generatedContent.data, image_data: result.image_data, results: generatedContent.data.results } })
      } else if (generatedContent.type === "service") {
        data.append("gen_image", "true")
        result = await generateImageService(data, apiUrl, apiKey)
        setGeneratedContent({ ...generatedContent, data: { ...generatedContent.data, image: result.image, results: generatedContent.data.results } })
      } else {
        return
      }
      toast({ title: "Image regenerated!", description: "Ad image has been regenerated." })
    } catch (e) {
      toast({ title: "Error", description: "Failed to regenerate image.", variant: "destructive" })
    } finally {
      setIsLoading?.(false)
    }
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="bg-gradient-to-r from-purple-900 to-blue-900 p-4 shadow-md">
        <Tabs defaultValue="preview" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-1 bg-gray-800/50">
            <TabsTrigger
              value="preview"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-blue-500 data-[state=active]:text-white"
            >
              <ImageIcon size={16} />
              Preview
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-auto">
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
                    <>
                      <img
                        src={`data:image/png;base64,${showNoText ? generatedContent.data.image.image_no_text_base64 : generatedContent.data.image.image_base64}`}
                        alt={showNoText ? "No Text" : "With Text"}
                        className="w-full h-auto"
                      />
                      {generatedContent.data.image?.image_no_text_base64 && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="mt-2"
                          onClick={() => setShowNoText((v) => !v)}
                        >
                          {showNoText ? "Xem ảnh có chữ" : "Xem ảnh không chữ"}
                        </Button>
                      )}
                    </>
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
                      // Download image
                      let imgSrc = null
                      if (generatedContent?.type === "product" && generatedContent.data.image_data) {
                        imgSrc = generatedContent.data.image_data
                      } else if (generatedContent?.type === "service" && generatedContent.data.image?.image_base64) {
                        imgSrc = showNoText && generatedContent.data.image?.image_no_text_base64
                          ? generatedContent.data.image.image_no_text_base64
                          : generatedContent.data.image.image_base64
                      }
                      if (imgSrc) {
                        const link = document.createElement("a")
                        link.href = `data:image/png;base64,${imgSrc}`
                        link.download = "facebook-ad.png"
                        document.body.appendChild(link)
                        link.click()
                        document.body.removeChild(link)
                        toast({
                          title: "Image downloaded!",
                          description: "Your Facebook ad image has been downloaded.",
                        })
                      }
                    }}
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
      </div>
    </div>
  )
}
