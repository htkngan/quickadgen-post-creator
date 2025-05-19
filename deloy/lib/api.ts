// Replace with your actual API key and URL
export const API_KEY = "your-api-key"
export const API_URL = "http://localhost:8000"

export async function generateTextAd(formData: FormData) {
  try {
    const response = await fetch(`${API_URL}/generate-ad`, {
      method: "POST",
      headers: {
        "x-api-key": API_KEY,
      },
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error("Error generating text ad:", error)
    throw error
  }
}

export async function generateServiceAd(formData: FormData) {
  try {
    const response = await fetch(`${API_URL}/generate-image-service`, {
      method: "POST",
      headers: {
        "x-api-key": API_KEY,
      },
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error("Error generating service ad:", error)
    throw error
  }
}

export async function generateProductAd(formData: FormData) {
  try {
    const response = await fetch(`${API_URL}/generate-product-ad`, {
      method: "POST",
      headers: {
        "x-api-key": API_KEY,
      },
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error("Error generating product ad:", error)
    throw error
  }
}
