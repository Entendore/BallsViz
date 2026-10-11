import sys
import os
import shutil
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QLineEdit, 
                               QFileDialog, QLabel, QSpinBox, QProgressBar)
from PySide6.QtCore import QThread, Signal, Qt
from moviepy.editor import ImageSequenceClip

# --- Simulation & Frame Rendering ---
class RippleSimulation:
    def __init__(self, width=1080, height=1920):
        self.W = width
        self.H = height
        self.gravity = 300.0
        self.restitution = 0.85
        self.friction = 0.99
        
        # Ripple parameters
        self.base_y = 1200
        self.amplitude = 250
        self.wavelength = 500.0
        self.omega = 1.5 # Speed of wave propagation
        
        # Fonts
        try:
            self.font_title = ImageFont.truetype("arial.ttf", 70)
            self.font_tab = ImageFont.truetype("arial.ttf", 40)
        except:
            self.font_title = ImageFont.load_default()
            self.font_tab = ImageFont.load_default()

    def init_balls(self, num_balls):
        self.N = num_balls
        self.X = np.random.uniform(100, self.W - 100, self.N)
        self.Y = np.random.uniform(500, 900, self.N)
        self.Vx = np.random.uniform(-50, 50, self.N)
        self.Vy = np.random.uniform(0, 50, self.N)
        
        # Assign Rainbow colors based on initial X position
        hues = (self.X - 100) / (self.W - 200)
        colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues])
        self.colors = (colors * 255).astype(int)

    def update_physics(self, dt, t):
        # Apply gravity
        self.Vy += self.gravity * dt
        
        # Update positions
        self.X += self.Vx * dt
        self.Y += self.Vy * dt
        
        # Apply mild air friction to Vx so they settle in troughs
        self.Vx *= self.friction
        
        # Boundary Collisions (Left/Right)
        mask_left = self.X < 0
        self.X[mask_left] = 0
        self.Vx[mask_left] = np.abs(self.Vx[mask_left]) * self.restitution
        
        mask_right = self.X > self.W
        self.X[mask_right] = self.W
        self.Vx[mask_right] = -np.abs(self.Vx[mask_right]) * self.restitution
        
        # Top boundary (keep them from flying off screen forever)
        mask_top = self.Y < 400
        self.Y[mask_top] = 400
        self.Vy[mask_top] = np.abs(self.Vy[mask_top]) * self.restitution
        
        # Ripple (Wavy Floor) Collision
        k = 2 * np.pi / self.wavelength
        floor_y = self.base_y + self.amplitude * np.sin(k * self.X + self.omega * t)
        
        mask_floor = self.Y > floor_y
        self.Y[mask_floor] = floor_y[mask_floor]
        
        # Bounce off floor
        self.Vy[mask_floor] = -np.abs(self.Vy[mask_floor]) * self.restitution
        
        # Wave pushes balls horizontally (simulating the moving ripple carrying the balls)
        slope = self.amplitude * k * np.cos(k * self.X[mask_floor] + self.omega * t)
        self.Vx[mask_floor] += slope * dt * 2.0

    def render_frame(self, t, active_tab_text):
        img = Image.new('RGB', (self.W, self.H), (0, 0, 0))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw Text Title
        title = "Light Trapped in a Ripple"
        bbox = draw.textbbox((0, 0), title, font=self.font_title)
        text_w = bbox[2] - bbox[0]
        draw.text(((self.W - text_w) / 2, 100), title, font=self.font_title, fill=(255, 255, 255))
        
        # Draw Tabs
        tabs = ["50 balls", "500 balls", "5000 balls"]
        tab_w = 250
        spacing = 30
        start_x = (self.W - (tab_w * 3 + spacing * 2)) / 2
        
        for i, tab in enumerate(tabs):
            x0 = start_x + i * (tab_w + spacing)
            y0 = 220
            x1 = x0 + tab_w
            y1 = y0 + 60
            
            is_active = (tab == active_tab_text)
            fill_color = (50, 50, 50, 200) if not is_active else (255, 255, 255, 50)
            outline_color = (120, 120, 120) if not is_active else (255, 255, 255)
            text_color = (150, 150, 150) if not is_active else (255, 255, 255)
            
            draw.rounded_rectangle([x0, y0, x1, y1], radius=15, fill=fill_color, outline=outline_color, width=2)
            
            bbox_tab = draw.textbbox((0, 0), tab, font=self.font_tab)
            text_w_tab = bbox_tab[2] - bbox_tab[0]
            draw.text((x0 + (tab_w - text_w_tab)/2, y0 + 10), tab, font=self.font_tab, fill=text_color)
            
        # Draw Ripple Floor (Rainbow Wave)
        k = 2 * np.pi / self.wavelength
        prev_x = 0
        prev_y = self.base_y + self.amplitude * np.sin(k * 0 + self.omega * t)
        
        for x in range(10, self.W + 10, 10):
            y = self.base_y + self.amplitude * np.sin(k * x + self.omega * t)
            hue = (x / self.W)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            r, g, b = int(r*255), int(g*255), int(b*255)
            
            # Draw thick line
            draw.line([(prev_x, prev_y), (x, y)], fill=(r, g, b), width=8)
            prev_x, prev_y = x, y
            
        # Draw Balls
        r = 8 if self.N <= 50 else (4 if self.N <= 500 else 2)
        for i in range(self.N):
            x = int(self.X[i])
            y = int(self.Y[i])
            color = tuple(self.colors[i])
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
            
        return np.array(img)

# --- Video Generation Thread ---
class VideoWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, mode, duration, output_path):
        super().__init__()
        self.mode = mode
        self.duration = duration
        self.output_path = output_path

    def run(self):
        try:
            temp_dir = "temp_frames"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            fps = 30
            total_frames = int(self.duration * fps)
            sim = RippleSimulation()
            
            # Define segments based on mode
            if "All 3" in self.mode:
                segments = [
                    (total_frames // 3, 50, "50 balls"),
                    (total_frames // 3, 500, "500 balls"),
                    (total_frames - 2 * (total_frames // 3), 5000, "5000 balls")
                ]
            else:
                count = int(self.mode.split()[0])
                segments = [(total_frames, count, self.mode)]
                
            frame_idx = 0
            t = 0.0
            dt = 1.0 / fps
            
            for num_frames, num_balls, tab_text in segments:
                sim.init_balls(num_balls)
                for _ in range(num_frames):
                    sim.update_physics(dt, t)
                    img_array = sim.render_frame(t, tab_text)
                    
                    # Save frame
                    img = Image.fromarray(img_array)
                    img.save(os.path.join(temp_dir, f"frame_{frame_idx:05d}.png"))
                    
                    frame_idx += 1
                    t += dt
                    
                    # Update progress (Rendering is 80% of the bar)
                    progress = int((frame_idx / total_frames) * 80)
                    self.progress.emit(progress)
                    
            # Compile video
            self.progress.emit(85)
            clip = ImageSequenceClip(temp_dir, fps=fps)
            self.progress.emit(95)
            
            clip.write_videofile(
                self.output_path, 
                fps=fps, 
                codec='libx264', 
                audio_codec='aac',
                logger=None
            )
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            self.progress.emit(100)
            self.finished.emit(f"Video successfully saved to:\n{self.output_path}")
            
        except Exception as e:
            self.error.emit(str(e))

# --- Main GUI ---
class ShortsSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Light Trapped in a Ripple - Simulator")
        self.resize(600, 400)
        
        self.worker = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("YouTube Shorts Physics Simulator")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Mode Selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Simulation Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["50 balls", "500 balls", "5000 balls", "All 3 (Sequential)"])
        self.mode_combo.setCurrentIndex(3) # Default to All 3
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Duration (s):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(3, 30)
        self.duration_spin.setValue(12) # 4s per simulation if All 3
        settings_layout.addWidget(self.duration_spin)
        layout.addLayout(settings_layout)

        # Output Button
        self.btn_output = QPushButton("Select Output Video File")
        self.btn_output.clicked.connect(self.select_output)
        layout.addWidget(self.btn_output)
        self.output_path = None

        # Generate Button
        self.btn_generate = QPushButton("Generate Simulation Short")
        self.btn_generate.setStyleSheet("background-color: #ff0000; color: white; font-size: 16px; padding: 10px;")
        self.btn_generate.clicked.connect(self.generate_video)
        layout.addWidget(self.btn_generate)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def select_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "ripple_simulation_short.mp4", "Video Files (*.mp4)")
        if file_path:
            if not file_path.endswith('.mp4'):
                file_path += '.mp4'
            self.output_path = file_path
            self.btn_output.setText(f"Output: {os.path.basename(file_path)}")

    def generate_video(self):
        if not self.output_path:
            self.status_label.setText("Error: Please select an output file path first.")
            return

        self.btn_generate.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Rendering physics simulation... This may take a minute.")
        
        # Setup and start worker thread
        self.worker = VideoWorker(
            self.mode_combo.currentText(), 
            self.duration_spin.value(),
            self.output_path
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_video_finished)
        self.worker.error.connect(self.on_video_error)
        self.worker.start()

    def on_video_finished(self, message):
        self.status_label.setText(message)
        self.btn_generate.setEnabled(True)

    def on_video_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        self.btn_generate.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShortsSimulatorApp()
    window.show()
    sys.exit(app.exec())