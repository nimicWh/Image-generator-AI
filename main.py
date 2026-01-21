import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QScrollArea, QGridLayout,
    QFileDialog, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QPixmap, QImage, QDrag
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

# --- Worker Thread for Progressive Generation ---
class ProgressiveVariationsThread(QThread):
    update_thumbnail = pyqtSignal(Image.Image, int)
    finished = pyqtSignal(list)

    def __init__(self, pipe, prompt, width, height, steps, guidance, num_variations=4, callback_steps=5):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.num_variations = num_variations
        self.callback_steps = callback_steps

    def run(self):
        images = [None] * self.num_variations

        # generate all variations one by one (diffusers supports callback)
        for idx in range(self.num_variations):
            def callback(step, timestep, latents):
                if step % self.callback_steps == 0:
                    img = self.pipe.numpy_to_pil(latents)[0]
                    self.update_thumbnail.emit(img, idx)

            with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
                img = self.pipe(
                    self.prompt,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance,
                    callback=callback,
                    callback_steps=self.callback_steps
                ).images[0]

            images[idx] = img
            self.update_thumbnail.emit(img, idx)

        self.finished.emit(images)

# --- Draggable Thumbnail Label ---
class DraggableLabel(QLabel):
    def __init__(self, image, index, parent=None):
        super().__init__(parent)
        self.image = image
        self.index = index
        self.setPixmap(AIImageGenerator.pil2pixmap(image).scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio))
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(str(self.index))
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        source_index = int(event.mimeData().text())
        target_index = self.index
        self.parent().parent().reorder_images(source_index, target_index)
        event.acceptProposedAction()

# --- Main GUI ---
class AIImageGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Generator Pro+ PDF Edition")
        self.setGeometry(50, 50, 1000, 900)
        self.pipe = None
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.prompt_history = []
        self.history_index = -1
        self.generated_images = []
        self.thumbnail_labels = []
        self.selected_image = None

        self.init_ui()
        self.load_model()

    def init_ui(self):
        layout = QVBoxLayout()

        # Prompt input
        self.prompt_input = QLineEdit()
        layout.addWidget(QLabel("Enter your text prompt:"))
        layout.addWidget(self.prompt_input)

        # Undo/Redo Buttons
        undo_redo_layout = QHBoxLayout()
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo_prompt)
        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(self.redo_prompt)
        undo_redo_layout.addWidget(undo_btn)
        undo_redo_layout.addWidget(redo_btn)
        layout.addLayout(undo_redo_layout)

        # Style Presets
        style_group = QGroupBox("Style Presets")
        style_layout = QHBoxLayout()
        self.styles = {
            "Art": "digital painting, trending on artstation",
            "Photo": "realistic photo, high detail",
            "Cartoon": "cartoon style, vibrant colors",
            "Anime": "anime style, colorful, detailed"
        }
        for name, style_prompt in self.styles.items():
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, s=style_prompt: self.apply_style(s))
            style_layout.addWidget(btn)
        style_group.setLayout(style_layout)
        layout.addWidget(style_group)

        # Sliders
        self.width_slider = self.create_slider("Width", 256, 1024, 512)
        layout.addLayout(self.width_slider['layout'])
        self.height_slider = self.create_slider("Height", 256, 1024, 512)
        layout.addLayout(self.height_slider['layout'])
        self.steps_slider = self.create_slider("Steps", 10, 100, 50)
        layout.addLayout(self.steps_slider['layout'])
        self.guidance_slider = self.create_slider("Guidance Scale", 1, 20, 7)
        layout.addLayout(self.guidance_slider['layout'])

        # Generate Button
        self.generate_btn = QPushButton("Generate Variations")
        self.generate_btn.clicked.connect(self.generate_variations)
        layout.addWidget(self.generate_btn)

        # Scrollable Thumbnail Grid
        self.scroll_area = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.grid_widget)
        layout.addWidget(self.scroll_area)

        # Full Image Display
        self.image_label = QLabel()
        self.image_label.setFixedSize(512, 512)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        # Save Buttons
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Selected Image")
        self.save_btn.clicked.connect(self.save_image)
        self.save_all_btn = QPushButton("Save All Variations")
        self.save_all_btn.clicked.connect(self.save_all_images)
        self.save_pdf_btn = QPushButton("Export as PDF Contact Sheet")
        self.save_pdf_btn.clicked.connect(self.export_pdf)
        self.save_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
        self.save_pdf_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.save_all_btn)
        save_layout.addWidget(self.save_pdf_btn)
        layout.addLayout(save_layout)

        self.setLayout(layout)

    def create_slider(self, label_text, min_val, max_val, default_val):
        layout = QHBoxLayout()
        label = QLabel(f"{label_text}: {default_val}")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(lambda val, l=label, t=label_text: l.setText(f"{t}: {val}"))
        layout.addWidget(QLabel(label_text))
        layout.addWidget(slider)
        layout.addWidget(label)
        return {'layout': layout, 'slider': slider, 'label': label}

    def apply_style(self, style_prompt):
        current = self.prompt_input.text().strip()
        self.prompt_input.setText(f"{style_prompt}, {current}" if current else style_prompt)

    def undo_prompt(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.prompt_input.setText(self.prompt_history[self.history_index])

    def redo_prompt(self):
        if self.history_index < len(self.prompt_history) - 1:
            self.history_index += 1
            self.prompt_input.setText(self.prompt_history[self.history_index])

    def load_model(self):
        self.generate_btn.setEnabled(False)
        QApplication.processEvents()
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        self.pipe = self.pipe.to(device)
        self.generate_btn.setEnabled(True)

    def generate_variations(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return

        if not self.prompt_history or self.prompt_history[-1] != prompt:
            self.prompt_history.append(prompt)
            self.history_index = len(self.prompt_history) - 1

        width = self.width_slider['slider'].value()
        height = self.height_slider['slider'].value()
        steps = self.steps_slider['slider'].value()
        guidance = self.guidance_slider['slider'].value()

        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
        self.save_pdf_btn.setEnabled(False)
        self.clear_grid()

        self.thread = ProgressiveVariationsThread(
            self.pipe, prompt, width, height, steps, guidance,
            num_variations=4, callback_steps=5
        )
        self.thread.update_thumbnail.connect(self.update_thumbnail)
        self.thread.finished.connect(self.select_best_image)
        self.thread.start()

    def clear_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.generated_images = []
        self.thumbnail_labels = []

    def update_thumbnail(self, image, index):
        if len(self.generated_images) <= index:
            self.generated_images.append(image)
        else:
            self.generated_images[index] = image

        if len(self.thumbnail_labels) <= index:
            label = DraggableLabel(image, index, parent=self.grid_widget)
            label.mousePressEvent = lambda e, img=image: self.show_full_image(img)
            self.thumbnail_labels.append(label)
            self.grid_layout.addWidget(label, index // 2, index % 2)
        else:
            self.thumbnail_labels[index].setPixmap(AIImageGenerator.pil2pixmap(image).scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio))

    def show_full_image(self, image):
        image.thumbnail((512, 512))
        self.image_label.setPixmap(AIImageGenerator.pil2pixmap(image))
        self.selected_image = image
        self.save_btn.setEnabled(True)

    def select_best_image(self, images):
        if not images:
            self.generate_btn.setEnabled(True)
            return
        prompt = self.prompt_input.text().strip()
        inputs = self.clip_processor(text=[prompt]*len(images), images=images, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        best_idx = outputs.logits_per_image.argmax().item()
        self.show_full_image(images[best_idx])
        self.save_all_btn.setEnabled(True)
        self.save_pdf_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)

    def reorder_images(self, source_index, target_index):
        # Swap images
        self.generated_images[source_index], self.generated_images[target_index] = self.generated_images[target_index], self.generated_images[source_index]
        # Swap thumbnails
        self.thumbnail_labels[source_index], self.thumbnail_labels[target_index] = self.thumbnail_labels[target_index], self.thumbnail_labels[source_index]
        # Update grid
        for idx, label in enumerate(self.thumbnail_labels):
            self.grid_layout.addWidget(label, idx // 2, idx % 2)
            label.index = idx

    def save_image(self):
        if self.selected_image:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if path:
                self.selected_image.save(path)

    def save_all_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save All Images")
        if folder:
            for i, img in enumerate(self.generated_images):
                img.save(os.path.join(folder, f"variation_{i+1}.png"))

    def export_pdf(self):
        if not self.generated_images:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF Contact Sheet", "", "PDF Files (*.pdf)")
        if not path:
            return

        # Create contact sheet (2x2)
        thumb_size = 512
        padding = 20
        font = ImageFont.load_default()
        num_images = len(self.generated_images)
        sheet_width = 2 * (thumb_size + padding) + padding
        sheet_height = 2 * (thumb_size + padding + 20) + padding  # extra 20 for prompt text

        sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
        for idx, img in enumerate(self.generated_images):
            x = padding + (idx % 2) * (thumb_size + padding)
            y = padding + (idx // 2) * (thumb_size + padding + 20)
            temp = img.copy()
            temp.thumbnail((thumb_size, thumb_size))
            sheet.paste(temp, (x, y))
            draw = ImageDraw.Draw(sheet)
            prompt_text = self.prompt_input.text()
            draw.text((x, y + temp.height + 2), prompt_text, fill=(0, 0, 0), font=font)

        sheet.save(path)

    @staticmethod
    def pil2pixmap(image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimage)


# --- Run App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIImageGenerator()
    window.show()
    sys.exit(app.exec())
