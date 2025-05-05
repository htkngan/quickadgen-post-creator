'use client';
import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

const API_URL = 'http://localhost:8000';

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
    itemCode: 'DEFAULT_CODE',
    itemName: '',
    description: '',
    serviceHours: '',
    price: '',
    gen_image: false,
  });
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const [imageResult, setImageResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults([]);
    setImageResult(null);

    try {
      const isAPIAvailable = await checkAPIStatus();
      if (!isAPIAvailable) {
        throw new Error('API server is not available. Please check if the server is running.');
      }

      const data = new FormData();
      data.append('itemCode', formData.itemCode);
      data.append('itemName', formData.itemName);
      data.append('description', formData.description);
      data.append('serviceHours', formData.serviceHours);
      data.append('price', formData.price);
      data.append('gen_image', String(formData.gen_image));
      if (formData.gen_image && imageFile) {
        data.append('image', imageFile);
      }

      const response = await fetch(`${API_URL}/generate-image-service`, {
        method: 'POST',
        body: data,
        mode: 'cors',
      });

      const resData = await response.json();
      if (!response.ok) {
        throw new Error(resData.message || `HTTP error! status: ${response.status}`);
      }
      if (resData.status === 'success') {
        setResults(resData.results || []);
        if (resData.image && resData.image.status === 'success' && resData.image.image_base64) {
          setImageResult(resData.image.image_base64);
        } else {
          setImageResult(null);
        }
      } else {
        throw new Error(resData.message || 'No content generated');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to connect to the server. Please check if the server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6 text-center">Ad Content Generator</h1>
      <form onSubmit={handleSubmit} className="space-y-4" encType="multipart/form-data">
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
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="gen_image"
            checked={formData.gen_image}
            onChange={(e) => {
              setFormData({...formData, gen_image: e.target.checked});
              if (!e.target.checked) {
                setImageFile(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }
            }}
          />
          <label htmlFor="gen_image">Tạo ảnh quảng cáo</label>
        </div>
        {formData.gen_image && (
          <div>
            <label className="block mb-2">Chọn ảnh (PNG, JPG...)</label>
            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setImageFile(e.target.files[0]);
                } else {
                  setImageFile(null);
                }
              }}
              className="w-full p-2 border rounded"
            />
          </div>
        )}
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
                  <span>Response Time: {result.time?.toFixed(2)}s</span>
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
      {imageResult && (
        <div className="mt-8 flex flex-col items-center">
          <h2 className="text-xl font-bold mb-2">Ảnh quảng cáo đã tạo</h2>
          <img
            src={`data:image/png;base64,${imageResult}`}
            alt="Generated Ad"
            className="max-w-md border rounded shadow"
          />
        </div>
      )}
    </div>
    </>
  );
}
