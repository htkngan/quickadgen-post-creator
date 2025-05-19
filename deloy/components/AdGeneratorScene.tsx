"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useThree } from "@react-three/fiber"
import { Text, Html, Float } from "@react-three/drei"
import { motion } from "framer-motion-3d"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { useToast } from "@/components/ui/use-toast"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import { ImageIcon, MessageSquare, ShoppingBag, Briefcase } from "lucide-react"

const API_KEY = "your-api-key" // Replace with your actual API key
const API_URL = "http://localhost:8000" // Replace with your actual API URL

interface AdGeneratorSceneProps {
  onGenerateContent: (content: any) => void
  setIsLoading: (loading: boolean) => void
}

export default function AdGeneratorScene({ onGenerateContent, setIsLoading }: AdGeneratorSceneProps) {
  const { camera } = useThree()
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
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  // Position camera to see the scene properly
  useEffect(() => {
    camera.position.set(0, 0, 5)
    camera.lookAt(0, 0, 0)
  }, [camera])

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
    if (!formData.itemName) {
      toast({
        title: "Missing information",
        description: "Please enter a product/service name",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)
    try {
      const data = new FormData()
      data.append("itemCode", formData.itemCode)
      data.append("itemName", formData.itemName)
      data.append("serviceHours", formData.serviceHours)
      data.append("description", formData.description)
      data.append("price", formData.price)

      let endpoint = ""
      let response

      switch (activePanel) {
        case "text":
          endpoint = `${API_URL}/generate-ad`
          break
        case "product":
          endpoint = `${API_URL}/generate-product-ad`
          data.append("pattern", formData.pattern)
          data.append("positions", formData.positions)
          if (selectedFile) {
            data.append("product_images", selectedFile)
          }
          break
        case "service":
          endpoint = `${API_URL}/generate-image-service`
          data.append("gen_image", "true")
          if (selectedFile) {
            data.append("image", selectedFile)
          }
          break
      }

      response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "x-api-key": API_KEY,
        },
        body: data,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      onGenerateContent({
        type: activePanel,
        data: result,
        formData: { ...formData },
      })
    } catch (error) {
      console.error("Error generating ad:", error)
      toast({
        title: "Error",
        description: "Failed to generate ad. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <group>
      {/* 3D Background Elements */}
      <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
        <mesh position={[-3, 2, -5]} rotation={[0.5, 0.5, 0]}>
          <octahedronGeometry args={[1.5, 0]} />
          <meshStandardMaterial color="#8b5cf6" wireframe />
        </mesh>
      </Float>

      <Float speed={1.2} rotationIntensity={0.3} floatIntensity={0.3}>
        <mesh position={[3, -2, -4]} rotation={[0.5, 0.5, 0]}>
          <dodecahedronGeometry args={[1.2, 0]} />
          <meshStandardMaterial color="#3b82f6" wireframe />
        </mesh>
      </Float>

      <Float speed={1} rotationIntensity={0.4} floatIntensity={0.4}>
        <mesh position={[4, 2, -6]} rotation={[0.5, 0.5, 0]}>
          <icosahedronGeometry args={[1, 0]} />
          <meshStandardMaterial color="#ec4899" wireframe />
        </mesh>
      </Float>

      {/* Main Interface */}
      <motion.group
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <Html position={[0, 0, 0]} transform>
          <Card className="w-[600px] h-[500px] overflow-auto bg-white/90 backdrop-blur-md shadow-xl">
            <CardContent className="p-6">
              <Tabs defaultValue="text" value={activePanel} onValueChange={setActivePanel}>
                <TabsList className="grid grid-cols-3 mb-4">
                  <TabsTrigger value="text" className="flex items-center gap-2">
                    <MessageSquare size={16} />
                    Text Only
                  </TabsTrigger>
                  <TabsTrigger value="product" className="flex items-center gap-2">
                    <ShoppingBag size={16} />
                    Product Ad
                  </TabsTrigger>
                  <TabsTrigger value="service" className="flex items-center gap-2">
                    <Briefcase size={16} />
                    Service Ad
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="text" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="itemName">Product/Service Name</Label>
                    <Input
                      id="itemName"
                      name="itemName"
                      value={formData.itemName}
                      onChange={handleInputChange}
                      placeholder="Enter product or service name"
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
                      />
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="product" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="itemName">Product Name</Label>
                    <Input
                      id="itemName"
                      name="itemName"
                      value={formData.itemName}
                      onChange={handleInputChange}
                      placeholder="Enter product name"
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
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Product Image (optional)</Label>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full"
                      >
                        <ImageIcon className="mr-2 h-4 w-4" />
                        {selectedFile ? selectedFile.name : "Upload Product Image"}
                      </Button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
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

                <TabsContent value="service" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="itemName">Service Name</Label>
                    <Input
                      id="itemName"
                      name="itemName"
                      value={formData.itemName}
                      onChange={handleInputChange}
                      placeholder="Enter service name"
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
                        className="w-full"
                      >
                        <ImageIcon className="mr-2 h-4 w-4" />
                        {selectedFile ? selectedFile.name : "Upload Template Image"}
                      </Button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>

              <div className="mt-6">
                <Button
                  onClick={handleGenerateAd}
                  className="w-full bg-gradient-to-r from-purple-600 to-blue-500 hover:from-purple-700 hover:to-blue-600"
                >
                  Generate Facebook Ad
                </Button>
              </div>
            </CardContent>
          </Card>
        </Html>
      </motion.group>

      {/* 3D Title */}
      <Text
        position={[0, 2.5, 0]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        font="/fonts/Geist_Bold.json"
      >
        Facebook Ad Generator
      </Text>

      {/* Floating 3D Icon */}
      <Float speed={1.5} rotationIntensity={0.4} floatIntensity={0.4}>
        <mesh position={[0, 3.5, 0]}>
          <sphereGeometry args={[0.5, 32, 32]} />
          <meshStandardMaterial color="#4267B2" />
        </mesh>
      </Float>
    </group>
  )
}
