from typing import Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
from openai import OpenAI
import asyncio
import time
import tiktoken  # Add this import for token counting
import os

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


#gpt_model = os.getenv("gpt_model")
gemni_model = "gemini-2.0-flash"

app = FastAPI()

# Update CORS middleware to be more permissive during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
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
    
    item = Item(
        itemCode=itemCode,
        itemName=itemName,
        serviceHours=serviceHours if serviceHours != 0 else None,
        description=description,
        price=price if price != 0 else None
    )

    logger.info(f"Received request for product: {item.itemName} with gen_image={gen_image}")
    
    if gen_image and image is None:
        raise HTTPException(status_code=400, detail="Image file is required when gen_image is True")
    # Tạo nội dung quảng cáo
    results = await asyncio.gather(
        run_gemini(item)
    )
    
    # Ghi log kết quả văn bản
    for result in results:
        if result["status"] == "success":
            logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
        else:
            logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
    
    # Nếu gen_image=True, kiểm tra ảnh đã được tải lên chưa
    if gen_image:
        if not image:
            logger.warning("Image generation requested but no image uploaded")
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Image generation requested but no image uploaded",
                    "results": results,
                    "image": None
                },
                status_code=400
            )
            
        logger.info("Processing image generation")
        image_bytes = await image.read()

        try:
            # Tạo ảnh nền với nội dung quảng cáo và hình ảnh
            image_context = generator.generate_clean_background(results[0]['ad_content'], image_bytes)            
            
            # Chuyển đổi thành base64 để trả về qua API
            img_io = io.BytesIO()
            image_context.image.save(img_io, format="PNG")
            img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
            
            image_result = {
                "status": "success",
                "image_base64": img_base64
            }
            logger.info("Image generation successful")
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
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
    else:
        # Chỉ trả về kết quả văn bản quảng cáo
        logger.info("Text-only ad generation requested")
        return JSONResponse(
            content={
                "status": "success",
                "results": results,
                "image": None
            },
            status_code=200
        )

# Example endpoint to retrieve item (for testing)
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Example endpoint to update item (for testing)
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.itemName, "item_id": item_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)