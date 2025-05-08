from typing import Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
from openai import OpenAI
import asyncio
import time
import tiktoken  # Add this import for token counting
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

# Load environment variables
load_dotenv()

client_gemini = genai.Client()
client_gpt = OpenAI()
generator = AdGenerator()

# Data models for ad generation
class AdResponse(BaseModel):
    id: str
    created_at: str
    image_data: str  # Base64 encoded image
    metadata: dict  # Metadata about the ad

# In-memory storage for generated ads
ad_database = {}


#gpt_model = os.getenv("gpt_model")
gemni_model = "gemini-2.0-flash"

app = FastAPI()

# Update CORS middleware to be more permissive during development
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins during development
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"]
# )

origins = [
    "https://quickadgen-post-creator-1pm7.vercel.app",
    "http://localhost:3000"  # nếu bạn test local frontend
]

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

# Pydantic model for input
class Item(BaseModel):
    itemCode: str
    itemName: str 
    serviceHours: Union[int, None] = None
    description: Union[str, None] = None
    price: Union[int, None] = None

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Ad Content Generator API"}

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

#Set up two LLM clients
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model("gemnigemini-2.0-flash-exp-image-generation")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

async def run_gemini(item: Item):
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
        token_count = count_tokens(formatted_text)
        end_time = time.time()
        logger.info(f"Successfully generated Gemini content of length: {len(formatted_text)}")
        logger.info(f"Gemini response text: {formatted_text}")
        
        return {
            "model": gemni_model,
            "status": "success",
            "ad_content": formatted_text,
            "total_tokens": token_count,
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

async def run_gpt(item: Item, gpt_model: str):
    start_time = time.time()
    try:
        prompt = f"""
        Bạn là một chuyên gia viết quảng cáo, nhiệm vụ của bạn là viết một quảng cáo hoàn chỉnh bằng tiếng Việt cho sản phẩm {item.itemName} để đăng tải ngay lên nền tảng Facebook. 
        Với mô tả sản phẩm như sau: {item.description if item.description else "Không có mô tả, sử dụng kiến thức chung về sản phẩm."}. Cùng một số thông tin kèm theo như số giờ dịch vụ {item.serviceHours} và giá {item.price}.
        Giọng điệu phải thuyết phục và tập trung vào việc thu hút sự chú ý nhanh chóng và truyền tải giá trị chỉ trong vài giây, sử dụng ngôn ngữ nói trực tiếp đến mong muốn và điểm khó khăn của đối tượng. 
        Mục tiêu là tạo ra một kết nối có ý nghĩa và thúc đẩy hành động ngay lập tức, cho dù đó là mua hàng, đăng ký hay tìm hiểu thêm về sản phẩm.
        """
        
        logger.info(f"Attempting to generate content for GPT with product: {item.itemName}")
        response = await asyncio.to_thread(
            client_gpt.chat.completions.create,
            model=gpt_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        if not response or not response.choices or not response.choices[0].message.content:
            logger.error(f"Invalid response from GPT: {response}")
            return {"model": gpt_model, "status": "error", "ad_content": None, "time": 0, "error": "Invalid response format"}
        
        formatted_text = response.choices[0].message.content.strip()
        token_count = count_tokens(formatted_text)
        end_time = time.time()
        logger.info(f"Successfully generated GPT content of length: {len(formatted_text)}")
        logger.info(f"GPT response text: {formatted_text}")
        
        return {
            "model": gpt_model,
            "status": "success",
            "ad_content": formatted_text,
            "total_tokens": token_count,
            "time": end_time - start_time
        }
    
    except Exception as e:
        end_time = time.time()
        logger.error(f"GPT error: {str(e)}")
        return {
            "model": gpt_model,
            "status": "error",
            "ad_content": None,
            "time": end_time - start_time,
            "error": str(e)
        }
        

@app.post("/generate-ad")
async def generate_ad(item: Item):
    logger.info(f"Received request for product: {item.itemName}")
    
    # Chạy hai model song song
    results = await asyncio.gather(
        run_gemini(item),
        run_gpt(item, "gpt-4.1-mini"),
        run_gpt(item, "gpt-4.1-nano")
    )
    
    # Ghi log kết quả tổng thể
    for result in results:
        if result["status"] == "success":
            logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
        else:
            logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
    
    # Trả về kết quả từ cả hai model
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
):
    # Bước 1: Khởi tạo đối tượng Item
    item = Item(
        itemCode=itemCode,
        itemName=itemName,
        serviceHours=serviceHours or None,
        description=description,
        price=price or None
    )

    # Bước 2: Sinh nội dung quảng cáo bằng GenAI
    results = await asyncio.gather(run_gemini(item))

    for result in results:
        if result["status"] == "success":
            logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
        else:
            logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")

    # Nếu không cần tạo ảnh thì chỉ trả về kết quả văn bản
    if not gen_image:
        return JSONResponse(
            content={
                "status": "success",
                "results": results,
                "image": None
            },
            status_code=200
        )

    # Bước 3: Tạo ảnh quảng cáo nếu gen_image=True
    try:
        ad_content = results[0].get("ad_content", "")
        image_bytes = await image.read() if image and hasattr(image, "read") else None

        # Tạo ảnh nền với/không với ảnh template
        image_context = generator.generate_background_service(item.itemName, ad_content, image_bytes)
        image_context_no_text = generator.generate_clean_background(image_context)

        # Chuyển đổi ảnh sang base64
        format = "PNG"
        if hasattr(image, "filename") and image.filename:
            ext = image.filename.split(".")[-1].upper()
            format = ext if ext in ["PNG", "JPEG", "JPG", "WEBP"] else "PNG"

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

@app.post("/generate-product-ad", response_model=AdResponse)
async def generate_product_ad(
    item_name: str = Form(...),
    pattern: str = Form("geometric"),
    product_images: Optional[List[UploadFile]] = File(None),
    positions: str = Form("center"),
    description: Optional[str] = Form(None),
):
    """Generate an advertisement with multiple products"""
    ad_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    product_info = {
        "itemName": item_name,
        "pattern": pattern,
        "positions": [positions.strip()],  # chỉ truyền 1 vị trí, dạng list 1 phần tử
    }
    
    metadata = {
        "itemName": item_name,
        "pattern": pattern,
        "positions": [positions.strip()],  # chỉ truyền 1 vị trí, dạng list 1 phần tử
    }
    
    if description:
        product_info["description"] = description
        metadata["description"] = description
    
    # Generate background
    context = generator.generate_background_product(product_info)
    background_image = context.image
    
    # Detect and remove text
    mask, text_detected = generator.detect_text_in_image(background_image)
    if text_detected:
        background_image = generator.remove_text_from_image(background_image, mask)
    
    # Process product images if provided
    if product_images and len(product_images) > 0:
        position_list = [pos.strip() for pos in positions.split(",")]
        if len(position_list) == 1 and len(product_images) > 1:
            position_list = position_list * len(product_images)
        
        metadata["positions"] = position_list
        metadata["productCount"] = len(product_images)
        
        # Convert uploaded files to PIL Images
        product_pil_images = []
        for product_image in product_images:
            content = await product_image.read()
            img = Image.open(io.BytesIO(content)).convert("RGBA")
            product_pil_images.append(img)
        
        # Add products to background
        background_image = generator.add_products_to_image(
            background_image,
            product_pil_images,
            position_list
        )
    
    # Convert final image to base64
    buffered = io.BytesIO()
    background_image.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode()
    
    # Store ad information
    ad_data = {
        "id": ad_id,
        "created_at": timestamp,
        "image_data": image_data,
        "metadata": metadata
    }
    
    ad_database[ad_id] = ad_data
    return ad_data

# Example endpoint to retrieve item (for testing)
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Example endpoint to update item (for testing)
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.itemName, "item_id": item_id}