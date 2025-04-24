'use client';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';


const API_URL = 'https://quickadgen-post-creator-3.onrender.com/';

const checkAPIStatus = async () => {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};

export default function Home() {
  const [formData, setFormData] = useState({
    itemCode: 'DEFAULT_CODE', // Add default value for required field
    itemName: '',
    description: '',
    serviceHours: '',
    price: ''
  });
  const [results, setResults] = useState<Array<{
    model: string;
    status: string;
    ad_content: string | null;
    total_tokens: number;
    time: number;
    error?: string;
  }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults([]);

    try {
      // Check API status first
      const isAPIAvailable = await checkAPIStatus();
      if (!isAPIAvailable) {
        throw new Error('API server is not available. Please check if the server is running.');
      }

      // Format data before sending
      const formattedData = {
        ...formData,
        serviceHours: formData.serviceHours ? parseInt(formData.serviceHours) : null,
        price: formData.price ? parseInt(formData.price) : null,
      };

      const response = await fetch(`${API_URL}/generate-ad`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formattedData),
        mode: 'cors',
      });

      const data = await response.json();
      console.log('Response data:', data); // Debug log

      if (!response.ok) {
        throw new Error(data.message || `HTTP error! status: ${response.status}`);
      }

      if (data.status === 'success' && data.results) {
        setResults(data.results);
      } else {
        throw new Error(data.message || 'No content generated');
      }
    } catch (error) {
      console.error('Fetch error:', error);
      setError(error instanceof Error ? 
        error.message : 
        'Failed to connect to the server. Please check if the server is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6 text-center">Ad Content Generator</h1>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-2">Mã sản phẩm</label>
          <input
            type="text"
            value={formData.itemCode}
            onChange={(e) => setFormData({...formData, itemCode: e.target.value})}
            className="w-full p-2 border rounded"
            required
          />
        </div>

        <div>
          <label className="block mb-2">Tên sản phẩm</label>
          <input
            type="text"
            value={formData.itemName}
            onChange={(e) => setFormData({...formData, itemName: e.target.value})}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        
        <div>
          <label className="block mb-2">Mô tả</label>
          <textarea
            value={formData.description}
            onChange={(e) => setFormData({...formData, description: e.target.value})}
            className="w-full p-2 border rounded"
            rows={4}
          />
        </div>
        
        <div>
          <label className="block mb-2">Thời lượng (phút)</label>
          <input
            type="text"
            value={formData.serviceHours}
            onChange={(e) => setFormData({...formData, serviceHours: e.target.value})}
            className="w-full p-2 border rounded"
          />
        </div>
        
        <div>
          <label className="block mb-2">Giá bán</label>
          <input
            type="text"
            value={formData.price}
            onChange={(e) => setFormData({...formData, price: e.target.value})}
            className="w-full p-2 border rounded"
          />
        </div>
        
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded"
          disabled={loading}
        >
          {loading ? 'Generating...' : 'Generate Ad'}
        </button>
      </form>

      {loading && (
        <div className="mt-4 text-center">
          <p>Generating ad content...</p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
      </div>
      <div className="mx-20 p-6">
      {results.length > 0 && (
        <div className="mt-8 flex flex-row gap-4">
          {results.map((result, index) => (
            <div key={index} className="flex-1 p-4 border rounded bg-white shadow">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">{result.model}</h2>
                <div className="text-sm text-gray-600 space-x-4">
                  <span>Response Time: {result.time.toFixed(2)}s</span>
                  <span>Tokens: {result.total_tokens}</span>
                </div>
              </div>
              {result.status === 'success' ? (
                <div className="whitespace-pre-wrap bg-gray-50 p-4 rounded">
                  <ReactMarkdown>{result.ad_content || ''}</ReactMarkdown>
                </div>
              ) : (
                <div className="text-red-500 p-4">
                  Error: {result.error || 'Unknown error'}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      </div>
    </>
  );
}
