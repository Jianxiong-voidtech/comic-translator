import os
import textwrap
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    source_text: str
    translated_text: str = ""


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a readable system or fallback font without crashing across OSes."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def create_sample_manga_panel(output_path: str) -> Tuple[Image.Image, List[TextRegion]]:
    """Create a clean B&W sample manga panel with a speech bubble for local testing."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Outer panel border
    draw.rectangle([24, 24, width - 24, height - 24], outline="black", width=4)

    # Simple stylized character on the left
    draw.ellipse([90, 140, 290, 360], fill="#f8fafc", outline="black", width=3)
    draw.ellipse([125, 175, 255, 315], fill="white", outline="black", width=2)
    draw.ellipse([155, 225, 173, 248], fill="black")
    draw.ellipse([207, 225, 225, 248], fill="black")
    draw.arc([172, 262, 208, 286], start=15, end=165, fill="black", width=3)
    draw.rounded_rectangle([115, 325, 265, 520], radius=24, fill="#f1f5f9", outline="black", width=3)

    # Speech bubble on the right
    bubble_box = [390, 120, 710, 420]
    draw.ellipse(bubble_box, fill="white", outline="black", width=3)
    # Bubble tail pointing toward character
    draw.polygon([(395, 320), (340, 355), (418, 365)], fill="white", outline="black")
    draw.line([(396, 322), (417, 363)], fill="white", width=4)

    # Vertical-style lines inside bubble (simulated OCR region)
    font_jp = load_font(34, bold=True)
    lines = ["本当に", "ありがとう"]
    start_y = 195
    for idx, line in enumerate(lines):
        draw.text((515, start_y + idx * 52), line, fill="black", font=font_jp)

    img.save(output_path)

    regions = [
        TextRegion(
            x1=445,
            y1=175,
            x2=665,
            y2=345,
            source_text="本当にありがとう",
        )
    ]
    return img, regions


def build_mask_image(base_img: Image.Image, regions: List[TextRegion], mask_path: str) -> Image.Image:
    """Generate a visual mask showing detected dialogue boxes to erase."""
    mask_view = base_img.copy()
    draw = ImageDraw.Draw(mask_view, "RGBA")
    for r in regions:
        draw.rectangle([r.x1, r.y1, r.x2, r.y2], fill=(239, 68, 68, 95), outline=(220, 38, 38, 255), width=3)
    mask_view.save(mask_path)
    return mask_view


def translate_regions(regions: List[TextRegion]) -> List[TextRegion]:
    """Translate text via OpenAI-compatible API if configured, else use deterministic mock."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    mock_dictionary = {
        "本当にありがとう": "Thank you so much!",
    }

    if not api_key:
        for r in regions:
            r.translated_text = mock_dictionary.get(r.source_text, f"[EN] {r.source_text}")
        return regions

    import requests

    for r in regions:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a manga translator. Keep translations concise so they fit inside speech bubbles.",
                },
                {
                    "role": "user",
                    "content": f"Translate this Japanese manga line into natural, concise English: {r.source_text}",
                },
            ],
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        r.translated_text = data["choices"][0]["message"]["content"].strip()

    return regions


def render_translated_panel(base_img: Image.Image, regions: List[TextRegion], output_path: str) -> Image.Image:
    """Inpaint detected text boxes with white and render wrapped translated dialogue."""
    rendered = base_img.copy()
    draw = ImageDraw.Draw(rendered)

    for r in regions:
        pad = 6
        # Step 2: Inpaint/clean text region inside bubble
        draw.rectangle([r.x1 - pad, r.y1 - pad, r.x2 + pad, r.y2 + pad], fill="white")

        # Step 4: Auto-wrap & render translated dialogue
        box_w = max(80, r.x2 - r.x1)
        box_h = max(60, r.y2 - r.y1)
        font_size = 28 if len(r.translated_text) <= 24 else 20
        font = load_font(font_size, bold=True)

        max_chars_per_line = max(10, int(box_w / (font_size * 0.58)))
        wrapped_lines = textwrap.wrap(r.translated_text, width=max_chars_per_line) or [r.translated_text]

        line_height = int(font_size * 1.3)
        total_text_h = len(wrapped_lines) * line_height
        current_y = r.y1 + max(4, (box_h - total_text_h) // 2)

        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            current_x = r.x1 + max(4, (box_w - line_w) // 2)
            draw.text((current_x, current_y), line, fill="black", font=font)
            current_y += line_height

    rendered.save(output_path)
    return rendered
