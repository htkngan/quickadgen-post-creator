"use client"

import { Suspense } from "react"
import { Canvas } from "@react-three/fiber"
import { Environment, PresentationControls } from "@react-three/drei"
import AdGeneratorScene from "@/components/AdGeneratorScene"

interface AdGeneratorCanvasProps {
  onGenerateContent: (content: any) => void
  setIsLoading: (loading: boolean) => void
}

export default function AdGeneratorCanvas({ onGenerateContent, setIsLoading }: AdGeneratorCanvasProps) {
  return (
    <div className="w-full h-[calc(100vh-64px)]">
      <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
        <Suspense fallback={null}>
          <PresentationControls
            global
            zoom={0.8}
            rotation={[0, 0, 0]}
            polar={[-Math.PI / 4, Math.PI / 4]}
            azimuth={[-Math.PI / 4, Math.PI / 4]}
          >
            <AdGeneratorScene onGenerateContent={onGenerateContent} setIsLoading={setIsLoading} />
          </PresentationControls>
          <Environment preset="city" />
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
        </Suspense>
      </Canvas>
    </div>
  )
}
