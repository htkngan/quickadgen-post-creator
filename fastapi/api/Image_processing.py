import io
import os
import re
import PIL
from PIL import Image
from google import genai
from google.genai import types
import cv2
import numpy as np
import torch
import tiktoken
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
    def to_bytes(self, format="PNG"):
        """Chuyển đổi ảnh thành dạng bytes"""
        if self.image is not None:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format=format)
            return img_byte_arr.getvalue()
        return None
        
    def from_bytes(self, image_bytes: bytes):
        """Tạo ImageContext từ dữ liệu bytes"""
        if image_bytes:
            self.buffer = io.BytesIO(image_bytes)
            self.image = Image.open(self.buffer)
        return self
    
    def get_base64(self, format="PNG"):
        """Chuyển đổi ảnh thành chuỗi base64"""
        import base64
        image_bytes = self.to_bytes(format)
        if image_bytes:
            return base64.b64encode(image_bytes).decode('utf-8')
        return None
        
    def from_base64(self, base64_string: str):
        """Tạo ImageContext từ chuỗi base64"""
        import base64
        if base64_string:
            image_bytes = base64.b64decode(base64_string)
            return self.from_bytes(image_bytes)
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
    
    @staticmethod
    def clean_text(text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002500-\U00002BEF"  # Chinese characters
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", flags=re.UNICODE)

        punctuation_pattern = re.compile(r"[.,!?;:“”\"\'‘’…—–\-–()\[\]{}<>•*~@#$%^&+=/\\|]")

        # Xoá emoji
        text = emoji_pattern.sub('', text)
        # Xoá dấu câu
        text = punctuation_pattern.sub('', text)

        return text.strip()
    
    #Set up two LLM clients
    def count_tokens(self, text: str, has_image: bool = False) -> int:
        """Count the number of tokens in a text string and optional image."""
        try:
            encoding = tiktoken.encoding_for_model("gemini-2.0-flash-exp-image-generation")
            text_tokens = len(encoding.encode(text))
            # Based on Gemini documentation, images typically use ~258 tokens
            image_tokens = 258 if has_image else 0
            return text_tokens + image_tokens
        except Exception as e:
            print(f"Token counting error: {e}")
            # Return an estimate if encoding fails
            return len(text.split()) + (258 if has_image else 0)

    def generate_background_service(self, itemName: str, ads_text: str, image_bytes: bytes | None, model: str = "gemini-2.0-flash-exp-image-generation") -> ImageContext:
        if image_bytes:
            img = PIL.Image.open(io.BytesIO(image_bytes))
            prompt = (
                f'Create a background to advertise a product in facebook: {itemName} with the content is {self.clean_text(ads_text)} with the style like {img} but more creative. '
                'The poster should have a creative pattern, vivid images, and be high quality, 16:9 ratio, '
                'style photographic, modern, impressive, creative.'
            )
            # Calculate tokens before API call
            token_count = self.count_tokens(prompt, has_image=True)
            print(f"Token count for request with image: {token_count}")
            
            response_image = self.genai_client.models.generate_content(
            model=model,
            contents=(prompt, img),
            config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # You can also get usage metrics from the response if available
            if hasattr(response_image, 'usage_metadata'):
                print(f"Actual token usage: {response_image.usage_metadata}")
                
            for part in response_image.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    return ImageContext(image=image)
            raise ValueError("No image returned from GenAI")
        else:
            prompt = (
                f'Create a background to advertise a product in facebook:{itemName} with the content is {self.clean_text(ads_text)}. '
                'The poster should have a creative pattern, vivid images, and be high quality, 16:9 ratio, '
                'style photographic, modern, impressive, creative.'
            )
            # Calculate tokens for text-only prompt
            token_count = self.count_tokens(prompt)
            print(f"Token count for text-only request: {token_count}")
            
            response_image = self.genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # Track usage metrics if available
            if hasattr(response_image, 'usage_metadata'):
                print(f"Actual token usage: {response_image.usage_metadata}")
                
            for part in response_image.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    return ImageContext(image=image)
            raise ValueError("No image returned from GenAI")

    def detect_bad_words(self, image_context: ImageContext):
        """
        Detects text in an image and generates a mask (không dùng file tạm)
        """
        if image_context.image is None:
            raise ValueError("ImageContext does not contain an image")
        img_pil = image_context.image.convert('RGB')
        img_np = np.array(img_pil)
        # PaddleOCR nhận numpy array (BGR)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        result = self.ocr.ocr(img_bgr, cls=True)
        if not result:
            return None, False
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        text_detected = False
        margin = 10
        for line in result:
            if not line:
                continue
            for word_info in line:
                text_detected = True
                points = np.array(word_info[0], dtype=np.int32)
                x_min = max(0, np.min(points[:, 0]) - margin)
                y_min = max(0, np.min(points[:, 1]) - margin)
                x_max = min(img_bgr.shape[1], np.max(points[:, 0]) + margin)
                y_max = min(img_bgr.shape[0], np.max(points[:, 1]) + margin)
                expanded_rect = np.array([
                    [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
                ], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [expanded_rect], color=255)
                original_points = points.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [original_points], color=255)
        if text_detected:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask, text_detected

    def generate_clean_background(self, image_context: ImageContext):
        """
        Nhận vào một ImageContext (ảnh đã có text), trả về ImageContext đã xóa text.
        """
        if not isinstance(image_context, ImageContext):
            raise TypeError("Input must be an ImageContext")
        mask, has_text = self.detect_bad_words(image_context)
        if not has_text:
            return image_context
        img_array = np.array(image_context.image.convert('RGB'))
        config = Config(
            ldm_steps=20,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )
        import time
        print(time.time())
        result = self.lama_model(img_array, mask, config)
        print(time.time())
        if result.dtype != np.uint8:
            if np.max(result) <= 1.0:
                result = (result * 255).astype(np.uint8)
            else:
                result = result.astype(np.uint8)
        cleaned_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return ImageContext(image=cleaned_image)

    def generate_prompt(self, item: dict) -> str:
        prompt = f"Tạo một poster quảng cáo vô cùng cao cấp và sáng tạo để đăng tải lên mạng xã hội facebook hình vuông cho {item['itemName']} tuân thủ quy tắc không hiển thị hình ảnh sản phẩm trong poster quảng cáo vì lí do bảo mật. "
        
        if 'description' in item and item['description']:
            prompt += f" Ảnh quảng cáo nên có chủ đề biểu diễn cho {item['description']}."
        if 'pattern' in item and item['pattern']:
            prompt += f" Ảnh quảng cáo cần có các chi tiết như {item['pattern']}. Nếu {item['pattern']} rỗng hãy cho nó phong cách thật high end, elegant, luxury, minimalist"
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

    def detect_text_in_image(self, image: Image.Image) -> tuple[np.ndarray, bool]:
        """Detect text in a PIL Image and return a mask"""
        # Convert PIL Image to numpy array for OCR
        img_np = np.array(image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        result = self.ocr.ocr(img_np, cls=True)
        if result is None:
            return np.zeros(img_np.shape[:2], dtype=np.uint8), False

        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        text_detected = False
        margin = 10

        for line in result:
            if line is None:
                continue
            for word_info in line:
                text_detected = True
                points = np.array(word_info[0], dtype=np.int32)
                
                x_min = max(0, np.min(points[:, 0]) - margin)
                y_min = max(0, np.min(points[:, 1]) - margin)
                x_max = min(img_np.shape[1], np.max(points[:, 0]) + margin)
                y_max = min(img_np.shape[0], np.max(points[:, 1]) + margin)
                
                expanded_rect = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.int32).reshape((-1, 1, 2))
                
                cv2.fillPoly(mask, [expanded_rect], color=255)
                original_points = points.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [original_points], color=255)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask, text_detected

    def remove_text_from_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Remove text from a PIL Image using a mask"""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        if mask.shape[:2] != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

        config = Config(
            ldm_steps=20,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )

        result = self.lama_model(image_np, mask, config)

        if result.dtype != np.uint8:
            if np.max(result) <= 1.0:
                result = (result * 255).astype(np.uint8)
            else:
                result = result.astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def process_text_in_image(self, image: Image.Image) -> Image.Image:
        """Process and remove text from an image if needed"""
        mask, text_detected = self.detect_text_in_image(image)
        if text_detected:
            return self.remove_text_from_image(image, mask)
        return image

    def add_products_to_image(self, background: Image.Image, products: list[Image.Image], positions: list[str]) -> Image.Image:
        """Add multiple products to a background image"""
        bg_image = background.copy()
        
        for i, product_image in enumerate(products):
            position = positions[i] if i < len(positions) else 'center'
            
            # Calculate target size
            original_ratio = product_image.width / product_image.height
            target_height = int(bg_image.height * 0.5)
            target_width = int(target_height * original_ratio)
            
            if target_width > bg_image.width:
                target_width = int(bg_image.width * 0.8)
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
                pos_x = int(bg_image.width * 0.1)
                pos_y = int(bg_image.height * 0.9) - product_image.height
            elif position == 'bottom-right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = int(bg_image.height * 0.9) - product_image.height
            else:
                raise ValueError("Invalid position")
            
            bg_image.paste(product_image, (pos_x, pos_y), product_image)
        
        return bg_image