"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Type, Palette, Save, Trash2, Undo, Redo, ZoomIn, ZoomOut } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { fabric } from "fabric"

interface FabricEditorProps {
  imageData: string
  initialText: string
}

export default function FabricEditor({ imageData, initialText }: FabricEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null)
  const [textColor, setTextColor] = useState("#ffffff")
  const [textSize, setTextSize] = useState(30)
  const [textShadow, setTextShadow] = useState(0)
  const [selectedFilter, setSelectedFilter] = useState("none")
  const [history, setHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const { toast } = useToast()

  // Save canvas state to history
  const saveToHistory = () => {
    if (!fabricCanvasRef.current) return

    const json = JSON.stringify(fabricCanvasRef.current.toJSON())

    // If we're not at the end of the history, remove everything after current index
    if (historyIndex < history.length - 1) {
      setHistory((prev) => prev.slice(0, historyIndex + 1))
    }

    setHistory((prev) => [...prev, json])
    setHistoryIndex((prev) => prev + 1)
  }

  // Undo action
  const undo = () => {
    if (historyIndex <= 0 || !fabricCanvasRef.current) return

    const newIndex = historyIndex - 1
    setHistoryIndex(newIndex)

    fabricCanvasRef.current.loadFromJSON(history[newIndex], () => {
      fabricCanvasRef.current?.renderAll()
    })
  }

  // Redo action
  const redo = () => {
    if (historyIndex >= history.length - 1 || !fabricCanvasRef.current) return

    const newIndex = historyIndex + 1
    setHistoryIndex(newIndex)

    fabricCanvasRef.current.loadFromJSON(history[newIndex], () => {
      fabricCanvasRef.current?.renderAll()
    })
  }

  useEffect(() => {
    if (!canvasRef.current) return

    // Initialize fabric canvas
    fabricCanvasRef.current = new fabric.Canvas(canvasRef.current, {
      width: 800,
      height: 800,
    })

    // Load the image
    const imageUrl = `data:image/png;base64,${imageData}`

    fabric.Image.fromURL(imageUrl, (img) => {
      // Scale image to fit canvas
      const scale = Math.min(
        fabricCanvasRef.current!.width! / img.width!,
        fabricCanvasRef.current!.height! / img.height!,
      )
      img.scale(scale)

      // Center the image
      img.set({
        left: fabricCanvasRef.current!.width! / 2,
        top: fabricCanvasRef.current!.height! / 2,
        originX: "center",
        originY: "center",
      })

      fabricCanvasRef.current!.add(img)
      fabricCanvasRef.current!.renderAll()

      // Add text from generated content
      if (initialText) {
        const text = new fabric.Textbox(initialText, {
          left: 50,
          top: 50,
          width: 700,
          fontSize: textSize,
          fill: textColor,
          fontFamily: "Arial",
          textAlign: "center",
          shadow: textShadow > 0 ? `rgba(0,0,0,0.6) ${textShadow}px ${textShadow}px ${textShadow}px` : "",
        })
        fabricCanvasRef.current!.add(text)
        fabricCanvasRef.current!.renderAll()
      }

      // Save initial state to history
      saveToHistory()

      // Add event listener for object modifications
      fabricCanvasRef.current!.on("object:modified", saveToHistory)
      fabricCanvasRef.current!.on("object:added", saveToHistory)
      fabricCanvasRef.current!.on("object:removed", saveToHistory)
    })

    return () => {
      fabricCanvasRef.current?.dispose()
    }
  }, [imageData, initialText])

  const applyFilter = (filter: string) => {
    if (!fabricCanvasRef.current) return

    // Get the background image (first object)
    const objects = fabricCanvasRef.current.getObjects()
    if (objects.length === 0) return

    const backgroundImage = objects[0] as fabric.Image

    // Remove existing filters
    backgroundImage.filters = []

    // Apply selected filter
    switch (filter) {
      case "grayscale":
        backgroundImage.filters.push(new fabric.Image.filters.Grayscale())
        break
      case "sepia":
        backgroundImage.filters.push(new fabric.Image.filters.Sepia())
        break
      case "brightness":
        backgroundImage.filters.push(new fabric.Image.filters.Brightness({ brightness: 0.1 }))
        break
      case "contrast":
        backgroundImage.filters.push(new fabric.Image.filters.Contrast({ contrast: 0.25 }))
        break
      case "saturation":
        backgroundImage.filters.push(new fabric.Image.filters.Saturation({ saturation: 0.3 }))
        break
      case "blur":
        backgroundImage.filters.push(new fabric.Image.filters.Blur({ blur: 0.25 }))
        break
    }

    backgroundImage.applyFilters()
    fabricCanvasRef.current.renderAll()
    setSelectedFilter(filter)
    saveToHistory()
  }

  const addText = () => {
    if (!fabricCanvasRef.current) return

    const text = new fabric.Textbox("Double click to edit", {
      left: 100,
      top: 100,
      width: 600,
      fontSize: textSize,
      fill: textColor,
      fontFamily: "Arial",
      textAlign: "center",
      shadow: textShadow > 0 ? `rgba(0,0,0,0.6) ${textShadow}px ${textShadow}px ${textShadow}px` : "",
    })

    fabricCanvasRef.current.add(text)
    fabricCanvasRef.current.setActiveObject(text)
    saveToHistory()
  }

  const updateTextProperties = () => {
    if (!fabricCanvasRef.current) return

    const activeObject = fabricCanvasRef.current.getActiveObject()
    if (activeObject && activeObject.type === "textbox") {
      const textbox = activeObject as fabric.Textbox
      textbox.set({
        fill: textColor,
        fontSize: textSize,
        shadow: textShadow > 0 ? `rgba(0,0,0,0.6) ${textShadow}px ${textShadow}px ${textShadow}px` : "",
      })
      fabricCanvasRef.current.renderAll()
      saveToHistory()
    }
  }

  const deleteSelectedObject = () => {
    if (!fabricCanvasRef.current) return

    const activeObject = fabricCanvasRef.current.getActiveObject()
    if (activeObject) {
      // Don't allow deleting the background image (first object)
      const objects = fabricCanvasRef.current.getObjects()
      if (objects.length > 0 && activeObject === objects[0]) {
        toast({
          title: "Cannot delete background",
          description: "The background image cannot be deleted.",
          variant: "destructive",
        })
        return
      }

      fabricCanvasRef.current.remove(activeObject)
      fabricCanvasRef.current.renderAll()
      saveToHistory()

      toast({
        title: "Object deleted",
        description: "The selected object has been removed.",
      })
    } else {
      toast({
        title: "No selection",
        description: "Please select an object to delete first.",
        variant: "destructive",
      })
    }
  }

  const saveImage = () => {
    if (fabricCanvasRef.current) {
      const dataURL = fabricCanvasRef.current.toDataURL({
        format: "png",
        quality: 1,
      })

      const link = document.createElement("a")
      link.download = "facebook-ad.png"
      link.href = dataURL
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      toast({
        title: "Image saved!",
        description: "Your Facebook ad has been downloaded successfully.",
      })
    }
  }

  const zoomIn = () => {
    if (!fabricCanvasRef.current) return

    const zoom = fabricCanvasRef.current.getZoom()
    fabricCanvasRef.current.setZoom(zoom * 1.1)
    fabricCanvasRef.current.renderAll()
  }

  const zoomOut = () => {
    if (!fabricCanvasRef.current) return

    const zoom = fabricCanvasRef.current.getZoom()
    fabricCanvasRef.current.setZoom(zoom / 1.1)
    fabricCanvasRef.current.renderAll()
  }

  useEffect(() => {
    updateTextProperties()
  }, [textColor, textSize, textShadow])

  return (
    <div className="flex flex-col lg:flex-row gap-6 p-6 animate-in fade-in-50 slide-in-from-bottom-10 duration-500">
      <div className="lg:w-3/4">
        <div className="bg-white rounded-lg shadow-xl p-4 transform-gpu hover:scale-[1.01] transition-transform duration-300">
          <div className="flex justify-between mb-4 bg-gray-100 p-2 rounded-md">
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={undo}
                disabled={historyIndex <= 0}
                className="hover:bg-gray-200"
              >
                <Undo className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={redo}
                disabled={historyIndex >= history.length - 1}
                className="hover:bg-gray-200"
              >
                <Redo className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={zoomIn} className="hover:bg-gray-200">
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={zoomOut} className="hover:bg-gray-200">
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={deleteSelectedObject}
                className="hover:bg-red-100 text-red-600 hover:text-red-700"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <div className="overflow-auto border border-gray-200 rounded-md">
            <canvas ref={canvasRef} width={800} height={800} className="mx-auto" />
          </div>
        </div>
      </div>

      <div className="lg:w-1/4">
        <Card className="shadow-xl transform-gpu hover:scale-[1.02] transition-transform duration-300 bg-gradient-to-br from-white to-purple-50">
          <CardContent className="p-4 space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-2 flex items-center text-purple-800">
                <Type className="mr-2 h-5 w-5" />
                Text Options
              </h3>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="textColor">Text Color</Label>
                  <div className="flex mt-1">
                    <input
                      type="color"
                      id="textColor"
                      value={textColor}
                      onChange={(e) => setTextColor(e.target.value)}
                      className="w-10 h-10 rounded cursor-pointer"
                    />
                    <input
                      type="text"
                      value={textColor}
                      onChange={(e) => setTextColor(e.target.value)}
                      className="ml-2 flex-1 px-3 py-2 border rounded-md hover:border-purple-500 focus:border-purple-500 transition-colors"
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="textSize">Text Size: {textSize}px</Label>
                  <Slider
                    id="textSize"
                    min={10}
                    max={100}
                    step={1}
                    value={[textSize]}
                    onValueChange={(value) => setTextSize(value[0])}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label htmlFor="textShadow">Text Shadow: {textShadow}px</Label>
                  <Slider
                    id="textShadow"
                    min={0}
                    max={10}
                    step={1}
                    value={[textShadow]}
                    onValueChange={(value) => setTextShadow(value[0])}
                    className="mt-2"
                  />
                </div>

                <Button
                  onClick={addText}
                  className="w-full bg-gradient-to-r from-purple-600 to-blue-500 hover:from-purple-700 hover:to-blue-600"
                >
                  Add Text
                </Button>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-2 flex items-center text-purple-800">
                <Palette className="mr-2 h-5 w-5" />
                Image Filters
              </h3>
              <Select value={selectedFilter} onValueChange={applyFilter}>
                <SelectTrigger className="hover:border-purple-500 focus:border-purple-500 transition-colors">
                  <SelectValue placeholder="Select filter" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="grayscale">Grayscale</SelectItem>
                  <SelectItem value="sepia">Sepia</SelectItem>
                  <SelectItem value="brightness">Brightness</SelectItem>
                  <SelectItem value="contrast">Contrast</SelectItem>
                  <SelectItem value="saturation">Saturation</SelectItem>
                  <SelectItem value="blur">Blur</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={saveImage}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-500 hover:from-purple-700 hover:to-blue-600 flex items-center justify-center gap-2"
            >
              <Save className="h-4 w-4" />
              Save Image
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
