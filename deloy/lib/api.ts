// API methods for the application
// This file contains functions to interact with the backend API
export async function generateAd(data: FormData, API_URL: string, apiKey: string) {
  const response = await fetch(`${API_URL}/generate-ad`, {
    method: 'POST',
    body: data,
    headers: {
      'x-api-key': apiKey
    },
  });
  const resData = await response.json();
  if (!response.ok) {
    throw new Error(resData.message || `HTTP error! status: ${response.status}`);
  }
  return resData;
}

export async function checkAPIStatus(API_URL: string) {
  const response = await fetch(`${API_URL}/health`);
  return response.ok;
}

export async function generateImageService(data: FormData, API_URL: string, apiKey: string) {
  const response = await fetch(`${API_URL}/generate-image-service`, {
    method: 'POST',
    body: data,
    headers: {
      'x-api-key': apiKey
    },
  });
  const resData = await response.json();
  if (!response.ok) {
    throw new Error(resData.message || `HTTP error! status: ${response.status}`);
  }
  return resData;
}

export async function generateProductAd(data: FormData, API_URL: string, apiKey: string) {
  const response = await fetch(`${API_URL}/generate-product-ad`, {
    method: 'POST',
    body: data,
    headers: {
      'x-api-key': apiKey
    },
  });
  const resData = await response.json();
  if (!response.ok) {
    throw new Error(resData.message || `HTTP error! status: ${response.status}`);
  }
  return resData;
}
