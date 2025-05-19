# app.py - FastAPI backend for Ad Content and Image Generation

from typing import Union
from typing import Annotated
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Security, Header, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import time
import os
import uuid
from datetime import datetime
import uvicorn
from Image_processing import ImageContext, AdGenerator
from google.genai import types
import base64
import io
from io import BytesIO
from PIL import Image
from typing import List, Optional
from model import Item, AdResponse
from fastapi.security import APIKeyHeader, APIKeyQuery

# Load environment variables from .env file
load_dotenv()

api_key_query = APIKeyQuery(name="api-key", auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

API_KEYS = os.environ.get("API_KEYS").split(",")

# Initialize Gemini client and AdGenerator
client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
generator = AdGenerator()

# In-memory storage for generated ads
ad_database = {}

# Model name for Gemini
# gpt_model = os.getenv("gpt_model")
gemni_model = "gemini-2.0-flash"

app = FastAPI()

# List of allowed origins (domains) for CORS requests to this API
origins = [
    "http://localhost:8000",
    "*"
]

# Add CORS middleware to allow requests from specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header)
) -> str:
    """
    Retrieve and validate API key from either query parameters or headers.

    This function checks if the provided API key (either in query parameters or headers)
    exists in the predefined list of valid API keys.

    Args:
        api_key_query (str): API key provided in the query parameters.
            Extracted using FastAPI's Security dependency with api_key_query.
        api_key_header (str): API key provided in the HTTP headers.
            Extracted using FastAPI's Security dependency with api_key_header.

    Returns:
        str: The valid API key.

    Raises:
        HTTPException: 401 Unauthorized error if neither API key is valid.
    """
    
    if api_key_query in API_KEYS:
        return api_key_query
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# Root endpoint
@app.get("/")
def read_root(x_api_key: Annotated[str, Header()] = None):
    """Root endpoint for API welcome message"""
    if get_api_key(x_api_key):
        return {"message": "Welcome to the Ad Content Generator API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok"}

async def run_gemini(item: Item):
    """
    Generate ad content using Gemini model for a given item
    """
    start_time = time.time()
    try:
        prompt = f"""
        Bạn là một chuyên gia viết quảng cáo, nhiệm vụ của bạn là viết một quảng cáo hoàn chỉnh bằng tiếng Việt cho sản phẩm {item.itemName} để đăng tải ngay lên nền tảng Facebook. Không cần giải thích nội dung.
        Với mô tả sản phẩm như sau: {item.description if item.description else "Không có mô tả, sử dụng kiến thức chung về sản phẩm."}. Cùng một số thông tin kèm theo như số giờ dịch vụ {item.serviceHours} và giá {item.price}.
        Giọng điệu phải thuyết phục và tập trung vào việc thu hút sự chú ý nhanh chóng và truyền tải giá trị chỉ trong vài giây, sử dụng ngôn ngữ nói trực tiếp đến mong muốn và điểm khó khăn của đối tượng. 
        Mục tiêu là tạo ra một kết nối có ý nghĩa và thúc đẩy hành động ngay lập tức, cho dù đó là mua hàng, đăng ký hay tìm hiểu thêm về sản phẩm.
        """
        
        logger.info(f"Attempting to generate content for Gemini with product: {item.itemName}")
        response = await asyncio.to_thread(
            client_gemini.models.generate_content,
            model=gemni_model,
            contents=[prompt]
        )
        
        if not response or not hasattr(response, 'text'):
            logger.error(f"Invalid response from Gemini: {response}")
            return {"model": gemni_model, "status": "error", "ad_content": None, "time": 0, "error": "Invalid response format"}
        
        formatted_text = response.text.strip()
        end_time = time.time()
        logger.info(f"Successfully generated Gemini content of length: {len(formatted_text)}")
        logger.info(f"Gemini response text: {formatted_text}")
        
        return {
            "model": gemni_model,
            "status": "success",
            "ad_content": formatted_text,
            "time": end_time - start_time
        }
    
    except Exception as e:
        end_time = time.time()
        logger.error(f"Gemini error: {str(e)}")
        return {
            "model": gemni_model,
            "status": "error",
            "ad_content": None,
            "time": end_time - start_time,
            "error": str(e)
        }    

@app.post("/generate-ad")
async def generate_ad(
    itemCode: str = Form(...),
    itemName: str = Form(...),
    serviceHours: Optional[int] = Form(0),
    description: Optional[str] = Form(None),
    price: Optional[int] = Form(0),
    x_api_key: Annotated[str, Header()] = None
):
    """
    Endpoint to generate ad content for a given item (using Form input)
    """
    if get_api_key(x_api_key):
        item = Item(
            itemCode=itemCode,
            itemName=itemName,
            serviceHours=serviceHours or None,
            description=description,
            price=price or None
        )
        logger.info(f"Received request for product: {item.itemName}")
        results = await asyncio.gather(
            run_gemini(item),
        )
        for result in results:
            if result["status"] == "success":
                logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
            else:
                logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
        return JSONResponse(
            content={
                "status": "success",
                "results": results
            },
            status_code=200
        )

@app.post("/generate-image-service")
async def generate_image_service(
    itemCode: str = Form(...),
    itemName: str = Form(...),
    serviceHours: Optional[int] = Form(0),
    description: Optional[str] = Form(None),
    price: Optional[int] = Form(0),
    image: Optional[UploadFile | str | None] = Form(None),
    gen_image: Optional[bool] = Form(False),
    x_api_key: Annotated[str, Header()] = None
):
    """
    Endpoint to generate ad content and optionally an ad image for a service
    """
    if get_api_key(x_api_key):
        item = Item(
            itemCode=itemCode,
            itemName=itemName,
            serviceHours=serviceHours or None,
            description=description,
            price=price or None
        )
        
        results = await asyncio.gather(run_gemini(item))
        for result in results:
            if result["status"] == "success":
                logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
            else:
                logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
        if not gen_image:
            return JSONResponse(
                content={
                    "status": "success",
                    "results": results,
                    "image": None
                },
                status_code=200
            )
        try:
            ad_content = results[0].get("ad_content", "")
            image_bytes = await image.read() if image and hasattr(image, "read") else None
            image_context = generator.generate_background_service(item.itemName, ad_content, image_bytes)
            image_context_no_text = generator.generate_clean_background(image_context)
            format = "PNG"
            if hasattr(image, "filename") and image.filename:
                ext = image.filename.split(".")[-1].upper()
                if ext == "JPG":
                    format = "JPEG"
                elif ext in ["PNG", "JPEG", "WEBP"]:
                    format = ext
                else:
                    format = "PNG"
            img_io = io.BytesIO()
            img_no_text_io = io.BytesIO()
            image_context.image.save(img_io, format=format)
            image_context_no_text.image.save(img_no_text_io, format=format)
            image_result = {
                "status": "success",
                "image_base64": base64.b64encode(img_io.getvalue()).decode("utf-8"),
                "image_no_text_base64": base64.b64encode(img_no_text_io.getvalue()).decode("utf-8")
            }
            logger.info("Image generation successful")
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}", exc_info=True)
            image_result = {
                "status": "failed",
                "error": str(e)
            }
        return JSONResponse(
            content={
                "status": "success",
                "results": results,
                "image": image_result
            },
            status_code=200
        )

@app.post("/generate-product-ad")
async def generate_product_ad(
    itemCode: str = Form(...),
    itemName: str = Form(...),
    pattern: str = Form("geometric"),
    product_images: Optional[List[UploadFile]] = File(None),
    positions: str = Form("center"),
    description: Optional[str] = Form(None),
    x_api_key: Annotated[str, Header()] = None
):
    """
    Generate a product advertisement image.

    Args:
        item_name (str): Name of the product.
        pattern (str): Visual pattern/style for the background.
        product_images (List[UploadFile], optional): List of product images to overlay.
        positions (str): Comma-separated positions for each product image.
        description (str, optional): Description of the product.

    Returns:
        AdResponse: Contains the ad ID, creation timestamp, base64-encoded image data.
    """
    if get_api_key(x_api_key):
        ad_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        item = Item(
            itemCode=itemCode,
            itemName=itemName,
            description=description,
        )
        
        print("Get_text_Ads")
        results = await asyncio.gather(run_gemini(item))
        for result in results:
            if result["status"] == "success":
                logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
            else:
                logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
        print(results)
                
        product_info = {
            "itemName": itemName,
            "pattern": pattern,
            "positions": [positions.strip()],
        }
        if description:
            product_info["description"] = description
        context = generator.generate_background_product(product_info)
        background_image = context.image
        mask, text_detected = generator.detect_text_in_image(background_image)
        
        if text_detected:
            background_image = generator.remove_text_from_image(background_image, mask)
            
        if product_images and len(product_images) > 0:
            position_list = [pos.strip() for pos in positions.split(",")]
            if len(position_list) == 1 and len(product_images) > 1:
                position_list = position_list * len(product_images)

            product_pil_images = []
            for product_image in product_images:
                content = await product_image.read()
                img = Image.open(io.BytesIO(content)).convert("RGBA")
                product_pil_images.append(img)
            background_image = generator.add_products_to_image(
                background_image,
                product_pil_images,
                position_list
            )
        buffered = io.BytesIO()
        background_image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()
        ad_data = {
            "id": ad_id,
            "results": results,
            "created_at": timestamp,
            "image_data": image_data,
        }
        ad_database[ad_id] = ad_data
        return ad_data

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None, x_api_key: Annotated[str, Header()] = None):
    """Get item by item_id (demo endpoint)"""
    if get_api_key(x_api_key):
        return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item, x_api_key: Annotated[str, Header()] = None):
    """Update item by item_id (demo endpoint)"""
    if get_api_key(x_api_key):
        return {"item_name": item.itemName, "item_id": item_id}

import uvicorn
if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)