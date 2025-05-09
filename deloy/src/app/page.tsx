'use client';
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

interface AdResult {
  model: string;
  status: 'success' | 'error';
  ad_content?: string;
  error?: string;
  time?: number;
  total_tokens?: number;
}

const API_URL = 'http://127.0.0.1:8000';

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
  const [apiType, setApiType] = useState<'generate-image-service' | 'generate-product-ad' | null>('generate-image-service');
  const [generationType, setGenerationType] = useState<'text-only' | 'with-image'>('with-image');
  const optionsContainerRef = useRef<HTMLDivElement>(null);

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
  useEffect(() => {
    function handleOutsideClick(event: MouseEvent) {
      if (optionsContainerRef.current && !optionsContainerRef.current.contains(event.target as Node)) {
        // Click was outside the options container, so deselect options
        setApiType(null);
      }
    }
    document.addEventListener('mousedown', handleOutsideClick);
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, []);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [results, setResults] = useState<AdResult[]>([]);
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
      {/* Navbar */}
      <nav className="bg-white dark:bg-gray-900 fixed w-full border-b h-20 border-gray-200 dark:border-gray-600 flex flex-row justify-between">
        <div className="flex flex-row w-full justify-end">
          <div className="flex flex-row my-5 mx-5 gap-2">
            <p className="text-white bg-blue-800 rounded-lg font-bold px-2 text-2xl text-center flex flex-row">
              <svg className="w-7 h-7 fill-[#ffffff] pt-2" viewBox="0 0 576 512" xmlns="http://www.w3.org/2000/svg">
                <path d="M234.7 42.7L197 56.8c-3 1.1-5 4-5 7.2s2 6.1 5 7.2l37.7 14.1L248.8 123c1.1 3 4 5 7.2 5s6.1-2 7.2-5l14.1-37.7L315 71.2c3-1.1 5-4 5-7.2s-2-6.1-5-7.2L277.3 42.7 263.2 5c-1.1-3-4-5-7.2-5s-6.1 2-7.2 5L234.7 42.7zM46.1 395.4c-18.7 18.7-18.7 49.1 0 67.9l34.6 34.6c18.7 18.7 49.1 18.7 67.9 0L529.9 116.5c18.7-18.7 18.7-49.1 0-67.9L495.3 14.1c-18.7-18.7-49.1-18.7-67.9 0L46.1 395.4zM484.6 82.6l-105 105-23.3-23.3 105-105 23.3 23.3zM7.5 117.2C3 118.9 0 123.2 0 128s3 9.1 7.5 10.8L64 160l21.2 56.5c1.7 4.5 6 7.5 10.8 7.5s9.1-3 10.8-7.5L128 160l56.5-21.2c4.5-1.7 7.5-6 7.5-10.8s-3-9.1-7.5-10.8L128 96 106.8 39.5C105.1 35 100.8 32 96 32s-9.1 3-10.8 7.5L64 96 7.5 117.2zm352 256c-4.5 1.7-7.5 6-7.5 10.8s3 9.1 7.5 10.8L416 416l21.2 56.5c1.7 4.5 6 7.5 10.8 7.5s9.1-3 10.8-7.5L480 416l56.5-21.2c4.5-1.7 7.5-6 7.5-10.8s-3-9.1-7.5-10.8L480 352l-21.2-56.5c-1.7-4.5-6-7.5-10.8-7.5s-9.1 3-10.8 7.5L416 352l-56.5 21.2z"></path>
              </svg>
              AI Post
            </p>
            
          </div>
        </div>
      </nav>
      {/* Sidebar */}
      <div className="fixed top-0 left-0 h-full w-auto bg-white shadow-lg z-40 border-r border-gray-200">
          <div className="p-4">
            <img 
              className="w-36 h-auto my-4" 
              src="https://s2-static-app.s3.ap-southeast-1.amazonaws.com/BadoSite/home/LOGO.svg" 
              alt="Bado Logo" 
              loading="lazy" 
            />
          </div>
          <div className="p-4">
            <ul className="space-y-3">
              <li className="p-2 hover:bg-blue-50 rounded-md transition-colors">
                <a href="#" className="flex items-center text-gray-700">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                  Dashboard
                </a>
              </li>
              <li className="p-2 hover:bg-blue-50 rounded-md transition-colors">
                <a href="#" className="flex items-center text-gray-700">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Analytics
                </a>
              </li>
              <li className="p-2 hover:bg-blue-50 rounded-md transition-colors">
                <a href="#" className="flex items-center text-gray-700">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  Projects
                </a>
              </li>
            </ul>
          </div>
        </div>
      {/* Main Content */}
      <div className="ml-64 pt-24 px-8 pb-8">
        {/*Tilte*/}
        <h1 className="text-2xl font-bold mb-6 text-center">AdGenius AI – Intelligent Content & Visual Creator for Smarter Advertising</h1>
        {/*Select option*/}
        <div className="mb-8 text-center" ref={optionsContainerRef}>
          <h3 className="text-lg font-medium text-gray-700 mb-4">Select the type of content you want to create</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Generate Image Service */}
            <div 
              className={`border rounded-lg p-6 cursor-pointer transition-all ${
                apiType === 'generate-image-service' 
                  ? 'border-blue-500 bg-blue-50 shadow-md' 
                  : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/30'
              }`}
              onClick={() => setApiType('generate-image-service')}
            >
              <div className="flex items-start">
                <div className="bg-blue-100 rounded-lg p-2 mr-4">
                  {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg> */}
                  <img 
                    src="https://www.svgrepo.com/show/530359/banana.svg" 
                    className="h-6 w-6 text-blue-700" 
                    alt="Image service icon"/>
                </div>
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <input
                      type="radio"
                      name="apiType"
                      value="generate-image-service"
                      checked={apiType === 'generate-image-service'}
                      onChange={() => setApiType('generate-image-service')}
                      className="h-4 w-4 text-blue-600"
                    />
                    <h4 className="font-semibold text-gray-800">Generate Image Service</h4>
                  </div>
                  <p className="text-gray-600 mt-2">
                    Design impactful ads with AI-generated images for your services
                  </p>
                </div>
              </div>
            </div>

            {/* Generate Product Ad */}
            <div 
              className={`border rounded-lg p-6 cursor-pointer transition-all ${
                apiType === 'generate-product-ad' 
                  ? 'border-blue-500 bg-blue-50 shadow-md' 
                  : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/30'
              }`}
              onClick={() => setApiType('generate-product-ad')}
            >
              <div className="flex items-start">
                <div className="bg-green-100 rounded-lg p-2 mr-4">
                <img 
                  src="https://www.svgrepo.com/show/530353/grape.svg" 
                  className="h-6 w-6 text-green-700" 
                  alt="Product icon" 
                />
                </div>
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <input
                      type="radio"
                      name="apiType"
                      value="generate-product-ad"
                      checked={apiType === 'generate-product-ad'}
                      onChange={() => setApiType('generate-product-ad')}
                      className="h-4 w-4 text-blue-600"
                    />
                    <h4 className="font-semibold text-gray-800">Generate Product Ad</h4>
                  </div>
                  <p className="text-gray-600 mt-2">
                    Create professional product advertisements with customizable positioning
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
         {/* Show sub-options only when a main option is selected */}
         {apiType && (
            <div className="border rounded-lg p-6 bg-gray-50 text-left mt-4" ref={optionsContainerRef}>
              <h4 className="font-medium text-gray-700 mb-3">Select generation type:</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Text Only Option */}
                <div 
                  className={`border rounded-lg p-4 cursor-pointer transition-all ${
                    generationType === 'text-only' 
                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                      : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/30'
                  }`}
                  onClick={() => setGenerationType('text-only')}
                >
                  <div className="flex items-center">
                    <div className="bg-yellow-100 rounded-lg p-2 mr-3">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                      </svg>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <input
                          type="radio"
                          name="generationType"
                          value="text-only"
                          checked={generationType === 'text-only'}
                          onChange={() => setGenerationType('text-only')}
                          className="h-4 w-4 text-blue-600"
                        />
                        <h5 className="font-medium text-gray-800">Text Only</h5>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        Generate advertisement text content only (/generate-ad)
                      </p>
                    </div>
                  </div>
                </div>
                               {/* With Image Option */}
                               <div 
                  className={`border rounded-lg p-4 cursor-pointer transition-all ${
                    generationType === 'with-image' 
                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                      : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/30'
                  }`}
                  onClick={() => setGenerationType('with-image')}
                >
                  <div className="flex items-center">
                    <div className="bg-purple-100 rounded-lg p-2 mr-3">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <input
                          type="radio"
                          name="generationType"
                          value="with-image"
                          checked={generationType === 'with-image'}
                          onChange={() => setGenerationType('with-image')}
                          className="h-4 w-4 text-blue-600"
                        />
                        <h5 className="font-medium text-gray-800">Text & Image</h5>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {apiType === 'generate-image-service' ? 
                          'Generate both text & image (/generate-image-service)' : 
                          'Generate both text & image (/generate-product-ad)'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-4 text-right">
                <button 
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
                  onClick={() => {
                    // Handle form submission based on selections
                    const endpoint = generationType === 'text-only' 
                      ? '/generate-ad'
                      : apiType // This will be either 'generate-image-service' or 'generate-product-ad'
                    
                    console.log(`Submitting to endpoint: ${endpoint}`)
                    // Proceed with form submission to the selected endpoint
                  }}
                >
                  Continue
                </button>
              </div>
            </div>
          )}

          <div className="flex flex-col md:flex-row gap-8">
            {/* Left: Form */}
            <div className="md:w-1/3 w-full">
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
