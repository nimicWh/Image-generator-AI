import sys, os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QSlider, QInputDialog, QGridLayout
)
from PyQt6.QtCore import Qt
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
import torch

# Constants
STYLE_COLORS = {"Art": (66, 135, 245), "Photo": (34, 177, 76),
                "Cartoon": (255, 127, 39), "Anime": (163, 73, 164)}
ICON_SIZE = (24, 24)

class AIImageGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Generator Pro")
        self.generated_images = []
        self.thumbnail_labels = []
        self.image_styles = []
        self.pipeline = self.load_pipeline()
        self.load_icons()
        self.init_ui()

    def load_pipeline(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        pipe.to(device)
        return pipe

    def load_icons(self):
        self.icons = {}
        for style in STYLE_COLORS.keys():
            icon_path = os.path.join("icons", f"{style.lower()}.png")
            if os.path.exists(icon_path):
                self.icons[style] = Image.open(icon_path).convert("RGBA").resize(ICON_SIZE)
            else:
                self.icons[style] = None

    def init_ui(self):
        layout = QVBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt here...")
        layout.addWidget(self.prompt_input)

        # Style buttons
        style_layout = QHBoxLayout()
        for style in STYLE_COLORS.keys():
            btn = QPushButton(style)
            btn.clicked.connect(lambda _, s=style: self.generate_images(s))
            style_layout.addWidget(btn)
        layout.addLayout(style_layout)

        # Grid for thumbnails
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)
        layout.addWidget(self.grid_widget)

        # Export button
        export_btn = QPushButton("Export as PDF")
        export_btn.clicked.connect(self.export_pdf)
        layout.addWidget(export_btn)

        self.setLayout(layout)

    def generate_images(self, style="Art"):
        prompt = self.prompt_input.text()
        if not prompt:
            return

        self.generated_images.clear()
        self.image_styles.clear()
        for i in range(4):  # 4 variations
            img = self.pipeline(prompt, guidance_scale=7.5).images[0]
            self.generated_images.append(img)
            self.image_styles.append(style)
            self.update_thumbnail(img, i, style)

    def update_thumbnail(self, image, index, style="Art"):
        img_with_number = image.copy()
        draw = ImageDraw.Draw(img_with_number)
        # Number overlay
        number_color = STYLE_COLORS.get(style, (255,255,255))
        rect_color = tuple(c//2 for c in number_color)
        rect_size = 30
        draw.rectangle([5,5,5+rect_size,5+rect_size], fill=rect_color+(200,))
        draw.text((10,5), f"#{index+1}", fill=number_color, font=ImageFont.load_default())
        # Style icon
        icon = self.icons.get(style)
        if icon: img_with_number.paste(icon, (img_with_number.width-ICON_SIZE[0]-5,5), mask=icon)

        if len(self.thumbnail_labels) <= index:
            label = QLabel()
            label.setPixmap(self.pil2pixmap(img_with_number).scaled(128,128,Qt.AspectRatioMode.KeepAspectRatio))
            self.thumbnail_labels.append(label)
            self.grid_layout.addWidget(label, index//2, index%2)
        else:
            self.thumbnail_labels[index].setPixmap(self.pil2pixmap(img_with_number).scaled(128,128,Qt.AspectRatioMode.KeepAspectRatio))

    @staticmethod
    def pil2pixmap(img):
        from PyQt6.QtGui import QPixmap, ImageQt
        return QPixmap.fromImage(ImageQt.ImageQt(img))

    def export_pdf(self):
        if not self.generated_images: return
        path,_ = QFileDialog.getSaveFileName(self,"Save PDF","","PDF Files (*.pdf)")
        if not path: return

        rows, ok1 = QInputDialog.getInt(self,"Grid Rows","Rows per page:",2,1,10)
        if not ok1: return
        cols, ok2 = QInputDialog.getInt(self,"Grid Columns","Columns per page:",2,1,10)
        if not ok2: return

        thumb_size = 512
        padding = 20
        text_space = 30
        legend_height = 40
        try: font_path = os.path.join("fonts","arial.ttf"); font=ImageFont.truetype(font_path,16)
        except: font=ImageFont.load_default()

        images_per_page = rows*cols
        pages = (len(self.generated_images)+images_per_page-1)//images_per_page
        pdf_pages = []
        page_width = cols*(thumb_size+padding)+padding
        page_height = rows*(thumb_size+padding+text_space)+padding+legend_height

        for page_num in range(pages):
            page_image = Image.new("RGB",(page_width,page_height),(255,255,255))
            for i in range(images_per_page):
                idx = page_num*images_per_page+i
                if idx>=len(self.generated_images): break
                img=self.generated_images[idx].copy(); style=self.image_styles[idx]; draw_img=ImageDraw.Draw(img)
                # Number overlay
                number_color=STYLE_COLORS.get(style,(255,255,255))
                rect_color=tuple(c//2 for c in number_color)
                rect_size=30
                draw_img.rectangle([5,5,5+rect_size,5+rect_size], fill=rect_color+(200,))
                draw_img.text((10,5), f"#{idx+1}", fill=number_color, font=font)
                # Style icon
                icon=self.icons.get(style)
                if icon: img.paste(icon, (img.width-ICON_SIZE[0]-5,5), mask=icon)

                # Paste thumbnail
                col=i%cols; row=i//cols
                x=padding+col*(thumb_size+padding); y=padding+row*(thumb_size+padding+text_space)
                page_image.paste(img,(x,y))

                # Prompt text
                draw_page=ImageDraw.Draw(page_image)
                prompt_text=self.prompt_input.text()
                text_width,_ = draw_page.textsize(prompt_text,font=font)
                text_x=x+(thumb_size-text_width)//2
                text_y=y+img.height+2
                draw_page.text((text_x,text_y), prompt_text, fill=(0,0,0), font=font)

            # Draw legend
            legend_y = page_height - legend_height + 5
            legend_x = padding; box_size=20; spacing=10
            for style_name,color in STYLE_COLORS.items():
                draw_page.rectangle([legend_x,legend_y,legend_x+box_size,legend_y+box_size],fill=color)
                draw_page.text((legend_x+box_size+5,legend_y), style_name, fill=(0,0,0), font=font)
                legend_x += box_size + 5 + draw_page.textsize(style_name,font=font)[0] + spacing

            pdf_pages.append(page_image)

        pdf_pages[0].save(path, save_all=True, append_images=pdf_pages[1:])

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=AIImageGenerator()
    win.show()
    sys.exit(app.exec())
