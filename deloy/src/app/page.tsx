'use client';
import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

const API_URL = 'https://quickadgen-post-creator-3.onrender.com';

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
  const [apiType, setApiType] = useState<'generate-image-service' | 'generate-product-ad'>('generate-image-service');
  const [formData, setFormData] = useState({
    itemCode: 'DEFAULT_CODE',
    itemName: '',
    description: '',
    serviceHours: '',
    price: '',
    gen_image: false,
    category: '',
    brand: '',
    pattern: '',
    positions: '',
  });
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageNoTextBase64, setImageNoTextBase64] = useState<string | null>(null);
  const [productAdImage, setProductAdImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults([]);
    setImageBase64(null);
    setImageNoTextBase64(null);
    setProductAdImage(null);

    try {
      const isAPIAvailable = await checkAPIStatus();
      if (!isAPIAvailable) {
        throw new Error('API server is not available. Please check if the server is running.');
      }

      const data = new FormData();
      if (apiType === 'generate-image-service') {
        data.append('itemCode', formData.itemCode);
        data.append('itemName', formData.itemName);
        data.append('description', formData.description);
        data.append('serviceHours', formData.serviceHours);
        data.append('price', formData.price);
        data.append('gen_image', String(formData.gen_image));
        if (formData.gen_image && imageFile) {
          data.append('image', imageFile);
        }
      } else if (apiType === 'generate-product-ad') {
        data.append('item_name', formData.itemName);
        data.append('pattern', formData.pattern || 'geometric');
        data.append('positions', formData.positions || 'center');
        data.append('description', formData.description);
        if (imageFile) {
          data.append('product_images', imageFile);
        }
      }

      const endpoint = apiType === 'generate-image-service' ? '/generate-image-service' : '/generate-product-ad';
      const response = await fetch(`${API_URL}${endpoint}`, {
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
        if (
          resData.image &&
          resData.image.status === 'success' &&
          (resData.image.image_base64 || resData.image.image_no_text_base64)
        ) {
          setImageBase64(resData.image.image_base64 || null);
          setImageNoTextBase64(resData.image.image_no_text_base64 || null);
          setProductAdImage(null);
        } else if (resData.image_data) {
          setProductAdImage(resData.image_data);
          setImageBase64(null);
          setImageNoTextBase64(null);
        } else {
          setImageBase64(null);
          setImageNoTextBase64(null);
          setProductAdImage(null);
        }
      } else if (resData.image_data) {
        setProductAdImage(resData.image_data);
        setImageBase64(null);
        setImageNoTextBase64(null);
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
      <nav className="bg-white dark:bg-gray-900 fixed w-full border-b h-15 border-gray-200 dark:border-gray-600 flex flex-row justify-between">
        <button className="mx-5" type="button" data-drawer-target="drawer-navigation" data-drawer-show="drawer-navigation" aria-controls="drawer-navigation">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
          </svg>
        </button>
        <div id="drawer-navigation" className="fixed top-0 left-0 z-40 w-64 h-screen p-4 overflow-y-auto transition-transform -translate-x-full bg-white dark:bg-gray-800" aria-labelledby="drawer-navigation-label">
          V
        </div>
        <div>
          <p className="text-right mt-3 px-5 text-2xl font-bold text-blue-800">
              Facebook
            </p>
        </div>
      </nav>
      <div className=" body max-w-6xl mx-auto p-6">
          <div className="flex flex-col md:flex-row gap-8">
            {/* Left: Form */}
            <div className="md:w-1/2 w-full">
              <h1 className="text-2xl font-bold mb-6 text-center md:text-left">Ad Content Generator</h1>
              <form onSubmit={handleSubmit} className="space-y-4" encType="multipart/form-data">
                <div className="flex gap-4 mb-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="apiType"
                      value="generate-image-service"
                      checked={apiType === 'generate-image-service'}
                      onChange={() => setApiType('generate-image-service')}
                    />
                    Generate Image Service
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="apiType"
                      value="generate-product-ad"
                      checked={apiType === 'generate-product-ad'}
                      onChange={() => setApiType('generate-product-ad')}
                    />
                    Generate Product Ad
                  </label>
                </div>
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
                  <label className="block mb-2">Giá bán</label>
                  <input
                    type="text"
                    value={formData.price}
                    onChange={(e) => setFormData({...formData, price: e.target.value})}
                    className="w-full p-2 border rounded"
                  />
                </div>
                {apiType === 'generate-product-ad' && (
                  <>
                    <div>
                      <label className="block mb-2">Pattern</label>
                      <input
                        type="text"
                        value={formData.pattern || ''}
                        onChange={(e) => setFormData({...formData, pattern: e.target.value})}
                        className="w-full p-2 border rounded"
                        placeholder="geometric"
                      />
                    </div>
                    <div>
                      <label className="block mb-2">Vị trí sản phẩm</label>
                      <select
                        value={formData.positions || 'center'}
                        onChange={(e) => setFormData({...formData, positions: e.target.value})}
                        className="w-full p-2 border rounded"
                      >
                        <option value="center">Center</option>
                        <option value="left">Left</option>
                        <option value="right">Right</option>
                        <option value="top">Top</option>
                        <option value="bottom">Bottom</option>
                        <option value="top-left">Top Left</option>
                        <option value="top-right">Top Right</option>
                        <option value="bottom-left">Bottom Left</option>
                        <option value="bottom-right">Bottom Right</option>
                      </select>
                    </div>
                    <div>
                      <label className="block mb-2">Chọn ảnh sản phẩm</label>
                      <div
                        className="border-2 border-dashed border-gray-400 rounded-lg p-4 flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 transition-colors"
                        onDragOver={e => { e.preventDefault(); e.stopPropagation(); }}
                        onDrop={e => {
                          e.preventDefault();
                          e.stopPropagation();
                          if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                            setImageFile(e.dataTransfer.files[0]);
                          }
                        }}
                        onClick={() => document.getElementById('product-image-upload')?.click()}
                        style={{ minHeight: 120 }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5-5m0 0l5 5m-5-5v12" /></svg>
                        <span className="text-gray-500 text-sm mb-2">Kéo thả hoặc bấm để chọn 1 ảnh (PNG, JPG...)</span>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => {
                            if (e.target.files && e.target.files.length > 0) {
                              setImageFile(e.target.files[0]);
                            } else {
                              setImageFile(null);
                            }
                          }}
                          className="hidden"
                          id="product-image-upload"
                        />
                        <label htmlFor="product-image-upload" className="bg-blue-100 text-blue-700 px-3 py-1 rounded cursor-pointer mt-2">Chọn ảnh</label>
                        {imageFile && (
                          <span className="mt-2 text-green-600 text-xs">Đã chọn: {imageFile.name}</span>
                        )}
                      </div>
                    </div>
                  </>
                )}
                {apiType === 'generate-image-service' && (
                  <>
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
                        <div
                          className="border-2 border-dashed border-gray-400 rounded-lg p-4 flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 transition-colors"
                          onDragOver={e => { e.preventDefault(); e.stopPropagation(); }}
                          onDrop={e => {
                            e.preventDefault();
                            e.stopPropagation();
                            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                              setImageFile(e.dataTransfer.files[0]);
                            }
                          }}
                          onClick={() => document.getElementById('ad-image-upload')?.click()}
                          style={{ minHeight: 120 }}
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5-5m0 0l5 5m-5-5v12" /></svg>
                          <span className="text-gray-500 text-sm mb-2">Kéo thả hoặc bấm để chọn 1 ảnh (PNG, JPG...)</span>
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
                            className="hidden"
                            id="ad-image-upload"
                          />
                          <label htmlFor="ad-image-upload" className="bg-blue-100 text-blue-700 px-3 py-1 rounded cursor-pointer mt-2">Chọn ảnh</label>
                          {imageFile && (
                            <span className="mt-2 text-green-600 text-xs">Đã chọn: {imageFile.name}</span>
                          )}
                        </div>
                      </div>
                    )}
                  </>
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

            {/* Right: Text Results */}
            <div className="md:w-1/2 w-full">
              {results.length > 0 && (
                <div className="w-full mt-0 flex flex-col gap-4">
                  {results.map((result, index) => (
                    <div key={index} className="p-4 border rounded bg-white shadow">
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
            </div>
          </div>

          {/* Bottom: Generated Images */}
          <div className="mt-8 w-full">
            {imageBase64 && imageNoTextBase64 && (
              <div className="flex flex-col items-center">
                <h2 className="text-2xl font-bold mb-4 text-center">Ảnh quảng cáo đã tạo</h2>
                <div className="flex flex-col md:flex-row gap-8 justify-center">
                  <div className="flex flex-col items-center">
                    <span className="mb-2 text-lg font-medium">Ảnh gốc</span>
                    <img
                      src={`data:image/png;base64,${imageBase64}`}
                      alt="Generated Ad"
                      className="max-w-md border rounded-lg shadow-lg"
                    />
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="mb-2 text-lg font-medium">Ảnh đã xoá chữ</span>
                    <img
                      src={`data:image/png;base64,${imageNoTextBase64}`}
                      alt="No Text Ad"
                      className="max-w-md border rounded-lg shadow-lg"
                    />
                  </div>
                </div>
              </div>
            )}
            {imageBase64 && !imageNoTextBase64 && (
              <div className="flex flex-col items-center">
                <h2 className="text-2xl font-bold mb-4 text-center">Ảnh quảng cáo đã tạo</h2>
                <img
                  src={`data:image/png;base64,${imageBase64}`}
                  alt="Generated Ad"
                  className="max-w-md border rounded-lg shadow-lg"
                />
              </div>
            )}
            {productAdImage && (
              <div className="flex flex-col items-center">
                <h2 className="text-2xl font-bold mb-4 text-center">Ảnh quảng cáo đã tạo</h2>
                <img
                  src={`data:image/png;base64,${productAdImage}`}
                  alt="Generated Product Ad"
                  className="max-w-md border rounded-lg shadow-lg"
                />
              </div>
            )}
          </div>
      </div>

    </>
  );
}
