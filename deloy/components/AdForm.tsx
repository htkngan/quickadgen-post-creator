"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { useToast } from "@/components/ui/use-toast"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import { ImageIcon, MessageSquare, ShoppingBag, Briefcase, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { generateAd, generateProductAd, generateImageService } from "@/lib/api"

interface AdFormProps {
  onGenerateContent: (content: any) => void
  setIsLoading: (loading: boolean) => void
  apiKey: string
  apiUrl: string
}

export default function AdForm({ onGenerateContent, setIsLoading, apiKey, apiUrl }: AdFormProps) {
  const [activePanel, setActivePanel] = useState("text")
  const [formData, setFormData] = useState({
    itemCode: "ITEM001",
    itemName: "",
    serviceHours: "0",
    description: "",
    price: "0",
    pattern: "geometric",
    positions: "center",
  })
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
    }
  }

  const handleGenerateAd = async () => {
    // Reset any previous error
    setApiError(null)

    if (!apiKey) {
      toast({
        title: "API Key Missing",
        description: "Vui lòng thiết lập API key trong phần cài đặt.",
        variant: "destructive",
      })
      return
    }

    if (!formData.itemName) {
      toast({
        title: "Thiếu thông tin",
        description: "Vui lòng nhập tên sản phẩm/dịch vụ",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)
    try {
      // Use real API
      const data = new FormData()
      data.append("itemCode", formData.itemCode)
      data.append("itemName", formData.itemName)
      data.append("serviceHours", formData.serviceHours)
      data.append("description", formData.description || "")
      data.append("price", formData.price)

      let apiCall: Promise<any> | null = null

      switch (activePanel) {
        case "text":
          apiCall = generateAd(data, apiUrl, apiKey)
          break
        case "product":
          data.append("pattern", formData.pattern)
          data.append("positions", formData.positions)
          if (selectedFile) {
            data.append("product_images", selectedFile)
          }
          apiCall = generateProductAd(data, apiUrl, apiKey)
          break
        case "service":
          data.append("gen_image", "true")
          if (selectedFile) {
            data.append("image", selectedFile)
          }
          apiCall = generateImageService(data, apiUrl, apiKey)
          break
      }

      try {
        if (!apiCall) throw new Error("No API call available.")
        const result = await apiCall
        onGenerateContent({
          type: activePanel,
          data: result,
          formData: { ...formData },
        })
      } catch (error) {
        if (error instanceof Error) {
          throw error
        }
        throw new Error("An unknown error occurred")
      }
    } catch (error) {
      console.error("Error generating ad:", error)

      let errorMessage = "Failed to generate ad. Please try again."

      if (error instanceof Error) {
        errorMessage = error.message

        // Set a more user-friendly error message for common issues
        if (errorMessage.includes("Failed to fetch") || errorMessage.includes("NetworkError")) {
          errorMessage = "Could not connect to the API server. Please check your internet connection and API URL."
          setApiError(errorMessage)
        } else if (errorMessage.includes("401")) {
          errorMessage = "Invalid API key. Please check your API key in settings."
          setApiError(errorMessage)
        } else if (errorMessage.includes("404")) {
          errorMessage = "API endpoint not found. Please check your API URL in settings."
          setApiError(errorMessage)
        } else {
          setApiError(errorMessage)
        }
      }

      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="w-[600px] max-w-[90vw] bg-white/90 backdrop-blur-md shadow-xl z-10 transform-gpu rotate-y-5 hover:rotate-y-0 transition-transform duration-500">
      <CardContent className="p-6">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-purple-800">Create Facebook Ads</h2>
          <p className="text-gray-600">Generate engaging ads with AI</p>
        </div>

        {apiError && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>API Connection Error</AlertTitle>
            <AlertDescription>{apiError}</AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="text" value={activePanel} onValueChange={setActivePanel}>
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger
              value="text"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-blue-500 data-[state=active]:text-white"
            >
              <MessageSquare size={16} />
              Text Only
            </TabsTrigger>
            <TabsTrigger
              value="product"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-blue-500 data-[state=active]:text-white"
            >
              <ShoppingBag size={16} />
              Product Ad
            </TabsTrigger>
            <TabsTrigger
              value="service"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-blue-500 data-[state=active]:text-white"
            >
              <Briefcase size={16} />
              Service Ad
            </TabsTrigger>
          </TabsList>

          <TabsContent value="text" className="space-y-4 animate-in fade-in-50 slide-in-from-left-5 duration-300">
            <div className="space-y-2">
              <Label htmlFor="itemName">Product/Service Name</Label>
              <Input
                id="itemName"
                name="itemName"
                value={formData.itemName}
                onChange={handleInputChange}
                placeholder="Enter product or service name"
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Enter product or service description"
                rows={3}
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="serviceHours">Service Hours</Label>
                <Input
                  id="serviceHours"
                  name="serviceHours"
                  type="number"
                  value={formData.serviceHours}
                  onChange={handleInputChange}
                  className="hover:border-purple-500 focus:border-purple-500 transition-colors"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="price">Price</Label>
                <Input
                  id="price"
                  name="price"
                  type="number"
                  value={formData.price}
                  onChange={handleInputChange}
                  className="hover:border-purple-500 focus:border-purple-500 transition-colors"
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="product" className="space-y-4 animate-in fade-in-50 slide-in-from-left-5 duration-300">
            <div className="space-y-2">
              <Label htmlFor="itemName">Product Name</Label>
              <Input
                id="itemName"
                name="itemName"
                value={formData.itemName}
                onChange={handleInputChange}
                placeholder="Enter product name"
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Enter product description"
                rows={3}
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="pattern">Visual Pattern</Label>
              <Input
                id="pattern"
                name="pattern"
                value={formData.pattern}
                onChange={handleInputChange}
                placeholder="e.g., geometric, abstract, minimal"
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="space-y-2">
              <Label>Product Image (optional)</Label>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full hover:bg-purple-100 transition-colors"
                >
                  <ImageIcon className="mr-2 h-4 w-4" />
                  {selectedFile ? selectedFile.name : "Upload Product Image"}
                </Button>
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="positions">Image Position</Label>
              <RadioGroup
                defaultValue="center"
                value={formData.positions}
                onValueChange={(value) => setFormData((prev) => ({ ...prev, positions: value }))}
                className="grid grid-cols-3 gap-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="top-left" id="top-left" />
                  <Label htmlFor="top-left">Top Left</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="top" id="top" />
                  <Label htmlFor="top">Top</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="top-right" id="top-right" />
                  <Label htmlFor="top-right">Top Right</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="left" id="left" />
                  <Label htmlFor="left">Left</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="center" id="center" />
                  <Label htmlFor="center">Center</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="right" id="right" />
                  <Label htmlFor="right">Right</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="bottom-left" id="bottom-left" />
                  <Label htmlFor="bottom-left">Bottom Left</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="bottom" id="bottom" />
                  <Label htmlFor="bottom">Bottom</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="bottom-right" id="bottom-right" />
                  <Label htmlFor="bottom-right">Bottom Right</Label>
                </div>
              </RadioGroup>
            </div>
          </TabsContent>

          <TabsContent value="service" className="space-y-4 animate-in fade-in-50 slide-in-from-left-5 duration-300">
            <div className="space-y-2">
              <Label htmlFor="itemName">Service Name</Label>
              <Input
                id="itemName"
                name="itemName"
                value={formData.itemName}
                onChange={handleInputChange}
                placeholder="Enter service name"
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Enter service description"
                rows={3}
                className="hover:border-purple-500 focus:border-purple-500 transition-colors"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="serviceHours">Service Hours</Label>
                <Input
                  id="serviceHours"
                  name="serviceHours"
                  type="number"
                  value={formData.serviceHours}
                  onChange={handleInputChange}
                  className="hover:border-purple-500 focus:border-purple-500 transition-colors"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="price">Price</Label>
                <Input
                  id="price"
                  name="price"
                  type="number"
                  value={formData.price}
                  onChange={handleInputChange}
                  className="hover:border-purple-500 focus:border-purple-500 transition-colors"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Template Image (optional)</Label>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full hover:bg-purple-100 transition-colors"
                >
                  <ImageIcon className="mr-2 h-4 w-4" />
                  {selectedFile ? selectedFile.name : "Upload Template Image"}
                </Button>
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <div className="mt-6">
          <Button
            onClick={handleGenerateAd}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-500 hover:from-purple-700 hover:to-blue-600 transform-gpu hover:scale-105 transition-all duration-300"
          >
            Generate Facebook Ad
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
