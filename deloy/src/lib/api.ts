interface AdRequest {
  itemCode: string;
  itemName: string;
  description?: string;
  serviceHours?: any;
  price?: any
}

export async function generateAd(data: AdRequest) {
  const response = await fetch('http://localhost:8000/generate-ad', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error('Failed to generate ad');
  }

  return response.json();
}
