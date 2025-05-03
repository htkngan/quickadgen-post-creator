import io
import os
from PIL import Image
from google import genai
from google.genai import types
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

# Import các module xử lý riêng
from paddleocr import PaddleOCR
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler

# Load env (nếu có API key cần thiết)
load_dotenv()

class ImageContext:
    def __init__(self, image: Image.Image = None, path: str = None):
        self.image = image            # Ảnh đang xử lý (PIL.Image)
        self.buffer = None            # Buffer RAM nếu cần
        self.path = path              # Path file lưu ảnh hiện tại
        self.metadata = {}            # Thông tin thêm (nếu cần)
    
    def load_from_path(self, path: str):
        self.path = path
        self.image = Image.open(path).convert("RGBA")
        self.buffer = None
        return self

    def save_to_path(self, path: str):
        if self.image is not None:
            self.image.save(path, format="PNG")
            self.path = path
        return self

    def to_buffer(self):
        if self.image is not None:
            buffer = io.BytesIO()
            self.image.save(buffer, format="PNG")
            buffer.seek(0)
            self.buffer = buffer
        return self

    def from_buffer(self, buffer: io.BytesIO):
        self.buffer = buffer
        self.image = Image.open(buffer)
        return self

class AdGenerator:
    def __init__(self):
        # Khởi tạo các model cần thiết
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
        self.lama_model = ModelManager(
            name="lama",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.genai_client = genai.Client()

    def generate_prompt(self, item: dict) -> str:
        """Generate a customized prompt for ad background generation based on provided parameters"""
        prompt = f"Tạo một poster quảng cáo vô cùng cao cấp và sáng tạo để đăng tải lên mạng xã hội facebook hình vuông cho sản phẩm {item['itemName']}, tuyệt đối không hiển thị hình ảnh sản phẩm vào poster quảng cáo vì lí do bảo mật. "
        
        # Add description if available
        if 'description' in item and item['description']:
            prompt += f" Ảnh quảng cáo nên có chủ đề biểu diễn cho {item['description']}."
        if 'pattern' in item and item['pattern']:
            prompt += f" Ảnh quảng cáo cần có các chi tiết như {item['pattern']}. Nếu {item['pattern']} rỗng hãy cho nó phong cách thật high end, elegant, luxury, minimalist"
            
        print(prompt)
        return prompt

    def generate_background_product(self, item: dict, model: str = "gemini-2.0-flash-exp-image-generation") -> ImageContext:
        prompt = self.generate_prompt(item)
        response_image = self.genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        for part in response_image.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
                return ImageContext(image=image)
        raise ValueError("No image returned from GenAI")
    
    def generate_background_service(self, item: dict, image_path: str, model: str = "gemini-2.0-flash-exp-image-generation") -> ImageContext:
        prompt = self.generate_prompt(item)
        img = cv2.imread(image_path)
        response_image = self.genai_client.models.generate_content(
            model=model,
            contents=(prompt, img),
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        for part in response_image.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
                return ImageContext(image=image)
        raise ValueError("No image returned from GenAI")


    def detect_bad_words(self, image_path: str, mask_output_path: str):
        img = cv2.imread(image_path)
        result = self.ocr.ocr(image_path, cls=True)
        if result is None:
           return mask_output_path, False

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        text_detected = False
        
        # Define expansion margin (in pixels)
        margin = 10  # Increase this value if text still isn't fully covered
        
        for line in result:
            if line is None:
                continue
            for word_info in line:
                text_detected = True
                points = np.array(word_info[0], dtype=np.int32)
                
                # Calculate bounding rectangle
                x_min = max(0, np.min(points[:, 0]) - margin)
                y_min = max(0, np.min(points[:, 1]) - margin)
                x_max = min(img.shape[1], np.max(points[:, 0]) + margin)
                y_max = min(img.shape[0], np.max(points[:, 1]) + margin)
                
                # Create expanded bounding rectangle
                expanded_rect = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.int32).reshape((-1, 1, 2))
                
                # Fill the expanded polygon
                cv2.fillPoly(mask, [expanded_rect], color=255)
                
                # Also fill the original detected polygon for better coverage
                original_points = points.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [original_points], color=255)

        # Optional: Apply dilation to further expand the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
                
        cv2.imwrite(mask_output_path, mask)
        return mask_output_path, text_detected

    def remove_text(self, image_path: str, mask_path: str, output_path: str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        config = Config(
            ldm_steps=20,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )

        result = self.lama_model(image, mask, config)

        if result.dtype != np.uint8:
            if np.max(result) <= 1.0:
                result = (result * 255).astype(np.uint8)
            else:
                result = result.astype(np.uint8)
                
        cv2.imwrite(output_path, result)

        return output_path

    def add_product_to_background(self, background_path: str, product_path: str, position: str, output_path: str, scale_factor: float = 0.5):
        bg_image = Image.open(background_path).convert("RGBA")
        product_image = Image.open(product_path).convert("RGBA")

        # Resize product image theo background giữ nguyên tỷ lệ ảnh gốc
        # Sản phẩm sẽ chiếm khoảng 1/2 chiều cao của ảnh nền
        original_ratio = product_image.width / product_image.height
        
        # Tính chiều cao mục tiêu (khoảng 1/2 chiều cao của bg)
        target_height = int(bg_image.height * 0.5)
        # Tính chiều rộng tương ứng để giữ nguyên tỷ lệ
        target_width = int(target_height * original_ratio)
        
        # Kiểm tra nếu chiều rộng lớn hơn width của bg, điều chỉnh lại
        if target_width > bg_image.width:
            target_width = int(bg_image.width * 0.8)  # Giới hạn tối đa 80% chiều rộng bg
            target_height = int(target_width / original_ratio)
            
        product_image = product_image.resize(
            (target_width, target_height), 
            Image.Resampling.LANCZOS
        )

        if position == 'left':
            pos = (0, (bg_image.height - product_image.height) // 2)
        elif position == 'right':
            pos = (bg_image.width - product_image.width, (bg_image.height - product_image.height) // 2)
        elif position == 'center':
            pos = ((bg_image.width - product_image.width) // 2, (bg_image.height - product_image.height) // 2)
        elif position == 'top':
            pos = ((bg_image.width - product_image.width) // 2, 0)
        elif position == 'bottom':
            pos = ((bg_image.width - product_image.width) // 2, bg_image.height - product_image.height)
        else:
            raise ValueError("Position must be one of ['left', 'right', 'center', 'top', 'bottom']")

        bg_image.paste(product_image, pos, product_image)
        bg_image.save(output_path, "PNG")
        return output_path

    def add_multiple_products(self, background_path: str, product_paths: list, positions: list, output_path: str, scale_factors: list = None):
        """Add multiple products to the background at different positions"""
        if not product_paths:
            raise ValueError("No product images provided")
        
        if positions and len(positions) != len(product_paths):
            raise ValueError("Number of positions must match number of products")
        
        if scale_factors and len(scale_factors) != len(product_paths):
            raise ValueError("Number of scale factors must match number of products")
        
        # Use default scale factor if not provided
        if not scale_factors:
            scale_factors = [0.5] * len(product_paths)
        
        # Use default position (center) if not provided
        if not positions:
            positions = ['center'] * len(product_paths)
        
        # Load background
        bg_image = Image.open(background_path).convert("RGBA")
        
        # Add each product
        for i, product_path in enumerate(product_paths):
            product_image = Image.open(product_path).convert("RGBA")
            position = positions[i]
            scale_factor = scale_factors[i]
            
            # Resize product image theo background giữ nguyên tỷ lệ ảnh gốc
            # Sản phẩm sẽ chiếm khoảng 1/2 chiều cao của ảnh nền
            original_ratio = product_image.width / product_image.height
            
            # Tính chiều cao mục tiêu (khoảng 1/2 chiều cao của bg)
            target_height = int(bg_image.height * 0.5)
            # Tính chiều rộng tương ứng để giữ nguyên tỷ lệ
            target_width = int(target_height * original_ratio)
            
            # Kiểm tra nếu chiều rộng lớn hơn width của bg, điều chỉnh lại
            if target_width > bg_image.width:
                target_width = int(bg_image.width * 0.8)  # Giới hạn tối đa 80% chiều rộng bg
                target_height = int(target_width / original_ratio)
                
            product_image = product_image.resize(
                (target_width, target_height), 
                Image.Resampling.LANCZOS
            )
            
            # Calculate position
            if position == 'left':
                pos_x = int(bg_image.width * 0.1)
                pos_y = (bg_image.height - product_image.height) // 2
            elif position == 'right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = (bg_image.height - product_image.height) // 2
            elif position == 'center':
                pos_x = (bg_image.width - product_image.width) // 2
                pos_y = (bg_image.height - product_image.height) // 2
            elif position == 'top':
                pos_x = (bg_image.width - product_image.width) // 2
                pos_y = int(bg_image.height * 0.1)
            elif position == 'bottom':
                pos_x = (bg_image.width - product_image.width) // 2
                pos_y = int(bg_image.height * 0.9) - product_image.height
            elif position == 'top-left':
                pos_x = int(bg_image.width * 0.1)
                pos_y = int(bg_image.height * 0.1)
            elif position == 'top-right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = int(bg_image.height * 0.1)
            elif position == 'bottom-left':
                pos_x = int(bg_image.width * 0.2)
                pos_y = int(bg_image.height * 0.9) - product_image.height
            elif position == 'bottom-right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = int(bg_image.height * 0.9) - product_image.height
            else:
                raise ValueError("Position must be one of ['left', 'right', 'center', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right']")
            
            bg_image.paste(product_image, (pos_x, pos_y), product_image)
        
        # Save final image
        bg_image.save(output_path, "PNG")
        return output_path
