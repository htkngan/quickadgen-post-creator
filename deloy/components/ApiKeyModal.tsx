"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface ApiKeyModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (apiKey: string, apiUrl: string) => void
  initialApiKey: string
  initialApiUrl: string
}

export default function ApiKeyModal({ isOpen, onClose, onSave, initialApiKey, initialApiUrl }: ApiKeyModalProps) {
  const [apiKey, setApiKey] = useState(initialApiKey)
  const [apiUrl, setApiUrl] = useState(initialApiUrl || "http://localhost:8000")

  const handleSave = () => {
    onSave(apiKey, apiUrl)
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px] bg-gradient-to-br from-gray-900 to-purple-900 text-white border-purple-500">
        <DialogHeader>
          <DialogTitle className="text-xl text-white">API Configuration</DialogTitle>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="apiKey" className="text-white col-span-4">
              API Key
            </Label>
            <Input
              id="apiKey"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              className="col-span-4 bg-gray-800 border-gray-700 text-white"
            />
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="apiUrl" className="text-white col-span-4">
              API URL
            </Label>
            <Input
              id="apiUrl"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="col-span-4 bg-gray-800 border-gray-700 text-white"
            />
          </div>
        </div>

        <DialogFooter>
          <Button
            onClick={handleSave}
            className="bg-gradient-to-r from-purple-600 to-blue-500 hover:from-purple-700 hover:to-blue-600"
          >
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
