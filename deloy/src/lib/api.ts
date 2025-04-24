interface AdRequest {
  itemCode: string;
  itemName: string;
  description?: string;
  serviceHours?: number;
  price?: number;
}

export async function generateAd(data: AdRequest) {
  const response = await fetch('https://quickadgen-post-creator-3.onrender.com/generate-ad', {
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
