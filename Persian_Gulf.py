import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QTabWidget, QTextEdit
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import arabic_reshaper
from bidi.algorithm import get_display

# Configuration
VIDEO_PATH = r"C:\Users\theal\Videos\Beauties of IRAN; The Persian Gulf coastline _ خلیج فارس، ایران.mp4"  # مسیر ویدیو را تنظیم کنید
FONT_PATH = "BNazanin.ttf"  # مسیر فونت را تنظیم کنید (یا DejaVuSans.ttf)
TEXT = "خلیج همیشه فارس"
FRAME_SIZE = (640, 480)
FPS = 15
INITIAL_DISPLAY_FRAMES = 45  # 3 seconds at 15 FPS for initial image display
INITIAL_DELAY_FRAMES = 30    # 3 seconds at 15 FPS for overlay and text
TRANSITION_FRAMES = 15       # 1 second transition at 15 FPS
VIDEO_START_SECONDS = 55     # Start video from 55 seconds
RAINBOW_COLORS = [
    (255, 0, 0),    # قرمز
    (255, 165, 0),  # نارنجی
    (255, 255, 0),  # زرد
    (0, 255, 0),    # سبز
    (0, 0, 255)     # آبی
]
COLOR_CHANGE_INTERVAL = 7.5  # 0.5 seconds at 15 FPS

# Define multiple blue ranges in HSV
LOWER_BLUE_1 = np.array([90, 60, 50])
UPPER_BLUE_1 = np.array([150, 255, 255])
LOWER_BLUE_2 = np.array([150, 40, 40])
UPPER_BLUE_2 = np.array([180, 255, 255])
LOWER_BLUE_3 = np.array([60, 50, 50])
UPPER_BLUE_3 = np.array([90, 255, 255])

# Check OpenCL
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
else:
    print("OpenCL is not supported or disabled")

def kmeans_color_clustering(frame, k=8):
    frame_np = frame if isinstance(frame, np.ndarray) else frame.get()
    pixels = frame_np.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    centers_reshaped = centers.reshape(1, -1, 3)
    hsv_centers = cv2.cvtColor(centers_reshaped, cv2.COLOR_BGR2HSV)[0]

    blue_indices = np.where(
        ((hsv_centers[:, 0] >= 60) & (hsv_centers[:, 0] <= 180)) & 
        (hsv_centers[:, 1] >= 40) & (hsv_centers[:, 2] >= 40)
    )[0]
    mask = np.zeros(frame_np.shape[:2], dtype=np.uint8)
    labels_2d = labels.reshape(frame_np.shape[:2])
    for idx in blue_indices:
        mask[labels_2d == idx] = 255

    return mask

def detect_blue_region(frame):
    frame = cv2.UMat(frame)

    if len(frame.get().shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.get().shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, cv2.UMat(LOWER_BLUE_1), cv2.UMat(UPPER_BLUE_1))
    mask2 = cv2.inRange(hsv, cv2.UMat(LOWER_BLUE_2), cv2.UMat(UPPER_BLUE_2))
    mask3 = cv2.inRange(hsv, cv2.UMat(LOWER_BLUE_3), cv2.UMat(UPPER_BLUE_3))

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    kmeans_mask = kmeans_color_clustering(frame.get())
    kmeans_mask = cv2.UMat(kmeans_mask)

    mask = cv2.bitwise_and(mask, kmeans_mask)

    kernel = cv2.UMat(np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 300:
            mask = cv2.UMat(np.zeros(mask.get().shape, dtype=np.uint8))
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask_3ch, len(contours) > 0

def create_animated_text_frame(frame_size, text, frame_count, delay_frames, blue_detected):
    img = Image.new('RGBA', frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 40)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 40)
        except IOError:
            font = ImageFont.load_default()

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    x = (frame_size[0] - text_size[0]) // 2
    y = 50 + 15 * np.sin(frame_count * 0.05)  # Text at top

    color_index = int(frame_count // COLOR_CHANGE_INTERVAL) % len(RAINBOW_COLORS)
    text_color = RAINBOW_COLORS[color_index]

    if frame_count >= delay_frames:
        scale = 1 + 0.15 * np.sin(frame_count * 0.08)
        font_size = int(40 * scale)
        try:
            font = ImageFont.truetype(FONT_PATH, font_size) if os.path.exists(FONT_PATH) else ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        x = (frame_size[0] - text_size[0]) // 2

        try:
            draw.text((x, y), bidi_text, font=font, fill=text_color)
        except Exception as e:
            print(f"Text rendering error: {e}")
            draw.text((x, y), bidi_text, font=font, fill=(255, 255, 255, 255))
    else:
        draw.text((x, y), bidi_text, font=font, fill=(255, 255, 255, 255))

    return np.array(img)

def create_notification_frame(frame_size, text, frame_count, duration_frames=60):
    img = Image.new('RGBA', frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    x = (frame_size[0] - text_size[0]) // 2
    y = 50

    alpha = 255
    if frame_count < 20:
        alpha = int(255 * (frame_count / 20))
    elif frame_count > duration_frames - 20:
        alpha = int(255 * ((duration_frames - frame_count) / 20))

    draw.text((x, y), bidi_text, font=font, fill=(255, 255, 255, alpha))
    return np.array(img)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️Persian Gulf❤️  @a_ra_80")
        self.setGeometry(100, 100, 700, 600)

        # Set RTL layout for the application
        QApplication.instance().setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 50))
        self.setPalette(palette)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a11cb, stop:1 #2575fc);
            border-bottom: 2px solid #1e3a8a;
            padding: 10px;
        """)
        title_label = QLabel("پوشش دیجیتال خلیج فارس")
        title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(header)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2575fc;
                background: #1e1e2f;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6a11cb, stop:1 #2575fc);
                color: white;
                padding: 10px;
                margin: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2575fc, stop:1 #6a11cb);
            }
        """)
        self.main_layout.addWidget(self.tab_widget)

        self.main_tab = QWidget()
        self.tab_widget.addTab(self.main_tab, "نمایش")
        main_tab_layout = QVBoxLayout(self.main_tab)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(FRAME_SIZE[0], FRAME_SIZE[1])
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #2575fc;")
        main_tab_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_layout = QHBoxLayout()
        main_tab_layout.addLayout(self.button_layout)

        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6a11cb, stop:1 #2575fc);
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 15px;
                border: none;
                margin: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2575fc, stop:1 #6a11cb);
                box-shadow: 0 0 15px #2575fc;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a8a, stop:1 #3b82f6);
            }
        """

        # Add buttons in reverse order to visually align RTL
        self.playback_btn = QPushButton("پردازش")
        self.playback_btn.setStyleSheet(button_style)
        self.playback_btn.clicked.connect(self.toggle_playback)
        self.button_layout.addWidget(self.playback_btn)

        self.toggle_source_btn = QPushButton("تغییر به وبکم")
        self.toggle_source_btn.setStyleSheet(button_style)
        self.toggle_source_btn.clicked.connect(self.toggle_source)
        self.button_layout.addWidget(self.toggle_source_btn)

        self.next_btn = QPushButton("تصویر بعدی")
        self.next_btn.setStyleSheet(button_style)
        self.next_btn.clicked.connect(self.show_next)
        self.button_layout.addWidget(self.next_btn)

        self.prev_btn = QPushButton("تصویر قبلی")
        self.prev_btn.setStyleSheet(button_style)
        self.prev_btn.clicked.connect(self.show_previous)
        self.button_layout.addWidget(self.prev_btn)

        self.select_folder_btn = QPushButton("انتخاب پوشه")
        self.select_folder_btn.setStyleSheet(button_style)
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.button_layout.addWidget(self.select_folder_btn)

        self.about_tab = QWidget()
        self.tab_widget.addTab(self.about_tab, "درباره")
        about_layout = QVBoxLayout(self.about_tab)
        about_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setStyleSheet("""
            QTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0a1e3a, stop:1 #3b1a6d);
                color: #ffffff;
                border: 3px solid #2575fc;
                border-radius: 15px;
                padding: 20px;
                font-size: 16px;
                font-family: 'BNazanin', 'DejaVu Sans', sans-serif;
                direction: rtl;
                text-align: right;
                box-shadow: 0 0 15px #2575fc;
            }
            QTextEdit:hover {
                border: 3px solid #6a11cb;
                box-shadow: 0 0 20px #6a11cb;
            }
        """)
        about_text.setText("""
            <h2 style='color: #6a11cb; font-size: 28px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); text-align: center;'>
                🌊 درباره نرم‌افزار پوشش خلیج فارس 🌊
            </h2>
            <p style='line-height: 1.8; color: #e0e0e0;'>
                <b style='color: #ffca28; font-size: 18px;'>🎨 هدف پروژه:</b><br>
                این نرم‌افزار با عشق به <span style='color: #2575fc; font-weight: bold;'>خلیج همیشه فارس</span> طراحی شده تا شکوه و زیبایی آن را به نمایش بگذارد. با بهره‌گیری از فناوری‌های پیشرفته پردازش تصویر، تصاویر و ویدیوهای شما با مناظر خلیج فارس ترکیب می‌شوند تا تجربه‌ای بی‌نظیر و خاطره‌انگیز خلق کنند.<br><br>
                <b style='color: #ffca28; font-size: 18px;'>🚀 ویژگی‌های کلیدی:</b><br>
                • 🌟 ترکیب هوشمند تصاویر و ویدیوها با مناظر خلیج فارس<br>
                • 📸 پشتیبانی از وبکم برای پردازش زنده و پویا<br>
                • 🎨 جلوه‌های بصری خیره‌کننده با رنگ‌های رنگین‌کمانی<br><br>
                <b style='color: #ffca28; font-size: 18px;'>👥 تیم توسعه:</b><br>
                ساخته شده توسط <span style='color: #00c851; font-weight: bold;'>Ali Rashidi (@a_ra_80)</span> با افتخار به فرهنگ و طبیعت ایران.<br><br>
                <b style='color: #ffca28; font-size: 18px;'>📬 ارتباط با ما:</b><br>
                سؤالی داری یا پیشنهادی به ذهنت رسیده؟ در تلگرام با ما در تماس باش: <a href='https://t.me/WriteYourWay' style='color: #40c4ff; text-decoration: none; font-weight: bold;'>t.me/WriteYourWay</a>
            </p>
            <p style='color: #ff6666; font-size: 18px; font-weight: bold; margin-top: 20px; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3); text-align: center;'>
                ❤️ با ما همراه شو تا زیبایی‌های خلیج فارس را به جهان نشان دهیم! ❤️
            </p>
        """)
        about_text.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        about_layout.addWidget(about_text)

        # Add Telegram button
        telegram_btn = QPushButton("📩 تماس در تلگرام")
        telegram_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2575fc, stop:1 #6a11cb);
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 12px;
                border: none;
                margin: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6a11cb, stop:1 #2575fc);
                box-shadow: 0 0 15px #2575fc;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a8a, stop:1 #3b82f6);
            }
        """)
        telegram_btn.clicked.connect(lambda: Qt.QDesktopServices.openUrl(QUrl("https://t.me/WriteYourWay")))
        about_layout.addWidget(telegram_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setSource(QUrl.fromLocalFile(VIDEO_PATH))

        self.using_webcam = False
        self.cap = None
        self.overlay_video = None
        self.current_file_index = 0
        self.files = []
        self.current_folder = ""
        self.current_frame = None
        self.frame_count = 0
        self.notification_text = "لطفاً یک پوشه انتخاب کنید."
        self.notification_frame_count = 0
        self.blue_detected = False
        self.blue_detected_frame = 0
        self.current_mask = None
        self.last_processed_frame = None
        self.transition_alpha = 0.0
        self.is_playing = False  # Default to stopped state

        self.initialize_folder()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle_playback(self):
        try:
            if self.is_playing:
                self.timer.stop()
                self.media_player.stop()
                if self.overlay_video:
                    self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))  # Reset to 55s
                self.media_player.setPosition(VIDEO_START_SECONDS * 1000)  # Reset audio to 55s
                self.frame_count = 0
                self.current_frame = None
                self.current_mask = None
                self.last_processed_frame = None
                self.blue_detected = False
                self.transition_alpha = 0.0
                # Clear the video label to show a black frame
                black_frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
                black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = black_frame_rgb.shape
                bytes_per_line = ch * w
                image = QImage(black_frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.video_label.setPixmap(pixmap)
                self.playback_btn.setText("پردازش")
                self.is_playing = False
                self.notification_text = "پردازش متوقف شد."
                self.notification_frame_count = 0
            else:
                # Reload current source to ensure frame is available
                if self.using_webcam:
                    self.initialize_webcam()
                elif self.current_folder:
                    self.initialize_folder()
                self.timer.start(1000 // FPS)
                self.playback_btn.setText("توقف")
                self.is_playing = True
                self.notification_text = "پردازش آغاز شد."
                self.notification_frame_count = 0
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def initialize_webcam(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.notification_text = "خطا: وبکم شناسایی نشد!"
                self.notification_frame_count = 0
                return
            if self.overlay_video:
                self.overlay_video.release()
            self.overlay_video = cv2.VideoCapture(VIDEO_PATH)
            if not self.overlay_video.isOpened():
                self.notification_text = "خطا: ویدیوی پوششی بارگذاری نشد!"
                self.notification_frame_count = 0
                self.cap.release()
                return
            self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))  # Start from 55s
            self.current_frame = None
            self.current_mask = None
            self.last_processed_frame = None
            self.blue_detected = False
            self.frame_count = 0
            self.transition_alpha = 0.0
            self.notification_text = "منبع به وبکم تغییر یافت."
            self.notification_frame_count = 0
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def initialize_folder(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = None
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.mp4')
            if self.current_folder:
                self.files = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith(image_extensions)])
                if not self.files:
                    self.notification_text = "خطا: هیچ فایل تصویری در پوشه یافت نشد!"
                    self.notification_frame_count = 0
                    return
                self.current_file_index = 0
                self.load_current_file()
                self.notification_text = "پوشه با موفقیت بارگذاری شد."
                self.notification_frame_count = 0
            else:
                self.current_frame = None
                self.current_mask = None
                self.last_processed_frame = None
                self.blue_detected = False
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def load_current_file(self):
        try:
            if not self.files:
                self.current_frame = None
                self.current_mask = None
                self.last_processed_frame = None
                self.blue_detected = False
                return
            file_path = os.path.join(self.current_folder, self.files[self.current_file_index])
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                frame = cv2.imread(file_path)
                if frame is None:
                    self.notification_text = f"خطا: فایل {os.path.basename(file_path)} بارگذاری نشد!"
                    self.notification_frame_count = 0
                    self.current_frame = None
                    self.current_mask = None
                    self.last_processed_frame = None
                    self.blue_detected = False
                    return
                self.current_frame = cv2.resize(frame, FRAME_SIZE)
                self.cap = None
                self.current_mask, self.blue_detected = detect_blue_region(self.current_frame)
                self.last_processed_frame = self.current_frame.copy()
                self.blue_detected_frame = self.frame_count
                self.frame_count = 0
                self.transition_alpha = 0.0
                self.notification_text = f"در حال نمایش: {os.path.basename(file_path)}"
                self.notification_frame_count = 0
            elif file_path.lower().endswith('.mp4'):
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    self.notification_text = f"خطا: ویدیوی {os.path.basename(file_path)} بارگذاری نشد!"
                    self.notification_frame_count = 0
                    self.current_frame = None
                    self.current_mask = None
                    self.last_processed_frame = None
                    self.blue_detected = False
                    return
                self.current_frame = None
                self.current_mask = None
                self.last_processed_frame = None
                self.blue_detected = False
                self.frame_count = 0
                self.transition_alpha = 0.0
                self.notification_text = f"در حال پخش: {os.path.basename(file_path)}"
                self.notification_frame_count = 0
            if self.overlay_video:
                self.overlay_video.release()
            self.overlay_video = cv2.VideoCapture(VIDEO_PATH)
            if not self.overlay_video.isOpened():
                self.notification_text = "خطا: ویدیوی پوششی بارگذاری نشد!"
                self.notification_frame_count = 0
                return
            self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))  # Start from 55s
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def select_folder(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "انتخاب پوشه تصاویر یا ویدیوها")
            if folder:
                self.current_folder = folder
                self.using_webcam = False
                self.toggle_source_btn.setText("تغییر به وبکم")
                self.initialize_folder()
                if self.is_playing:
                    self.frame_count = 0
                    self.media_player.stop()
                    self.media_player.setPosition(VIDEO_START_SECONDS * 1000)
                    self.transition_alpha = 0.0
            else:
                self.notification_text = "لغو: هیچ پوشه‌ای انتخاب نشد!"
                self.notification_frame_count = 0
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def show_previous(self):
        try:
            if not self.using_webcam and self.files:
                self.current_file_index = (self.current_file_index - 1) % len(self.files)
                self.load_current_file()
                if self.is_playing:
                    self.media_player.stop()  # Stop the audio
                    self.media_player.setPosition(VIDEO_START_SECONDS * 1000)  # Reset audio to 55s
                    if self.overlay_video:
                        self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))  # Reset video to 55s
                    self.frame_count = 0  # Reset frame count for 3-second delay
                    self.transition_alpha = 0.0  # Reset transition for overlay
                    self.current_mask = None  # Force re-detection of blue region
                    self.blue_detected = False  # Reset blue detection
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def show_next(self):
        try:
            if not self.using_webcam and self.files:
                self.current_file_index = (self.current_file_index + 1) % len(self.files)
                self.load_current_file()
                if self.is_playing:
                    self.media_player.stop()  # Stop the audio
                    self.media_player.setPosition(VIDEO_START_SECONDS * 1000)  # Reset audio to 55s
                    if self.overlay_video:
                        self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))  # Reset video to 55s
                    self.frame_count = 0  # Reset frame count for 3-second delay
                    self.transition_alpha = 0.0  # Reset transition for overlay
                    self.current_mask = None  # Force re-detection of blue region
                    self.blue_detected = False  # Reset blue detection
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def toggle_source(self):
        try:
            self.using_webcam = not self.using_webcam
            self.media_player.stop()
            if self.using_webcam:
                self.toggle_source_btn.setText("تغییر به پوشه")
                self.initialize_webcam()
            else:
                self.toggle_source_btn.setText("تغییر به وبکم")
                if self.current_folder:
                    self.initialize_folder()
                else:
                    self.select_folder()
            if self.is_playing:
                self.frame_count = 0
                self.media_player.setPosition(VIDEO_START_SECONDS * 1000)
                self.transition_alpha = 0.0
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def update_frame(self):
        try:
            if self.using_webcam or self.cap:
                if self.cap:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret:
                        self.notification_text = "خطا: فریم ویدیویی دریافت نشد!"
                        self.notification_frame_count = 0
                        return
                    frame = cv2.resize(frame, FRAME_SIZE)
                else:
                    self.notification_text = "خطا: منبع ویدیویی در دسترس نیست!"
                    self.notification_frame_count = 0
                    return
            else:
                if self.current_frame is None:
                    self.notification_text = "خطا: هیچ تصویری برای نمایش وجود ندارد!"
                    self.notification_frame_count = 0
                    return
                frame = self.current_frame.copy()

            self.frame_count += 1

            if self.using_webcam or self.cap:
                if self.last_processed_frame is None or np.mean(np.abs(frame - self.last_processed_frame)) > 10:
                    self.current_mask, self.blue_detected = detect_blue_region(frame)
                    self.last_processed_frame = frame.copy()
                    self.blue_detected_frame = self.frame_count
            else:
                self.current_mask, self.blue_detected = detect_blue_region(frame) if self.current_mask is None else (self.current_mask, self.blue_detected)

            if self.current_mask is None:
                self.notification_text = "خطا: ماسک منطقه آبی تولید نشد!"
                self.notification_frame_count = 0
                return

            result = frame.copy()

            # After 3 seconds, start overlay video and text
            if self.frame_count >= INITIAL_DISPLAY_FRAMES:
                if self.overlay_video and self.overlay_video.isOpened():
                    if self.frame_count == INITIAL_DISPLAY_FRAMES:
                        # Start video and audio at 3 seconds
                        self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))
                        self.media_player.setPosition(VIDEO_START_SECONDS * 1000)
                        self.media_player.play()

                    ret_overlay, overlay_frame = self.overlay_video.read()
                    if not ret_overlay:
                        self.overlay_video.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_SECONDS * int(self.overlay_video.get(cv2.CAP_PROP_FPS)))  # Loop to 55s
                        ret_overlay, overlay_frame = self.overlay_video.read()

                    overlay_frame = cv2.resize(overlay_frame, FRAME_SIZE)
                    overlay_frame = cv2.UMat(overlay_frame)
                    frame = cv2.UMat(frame)

                    if overlay_frame.get().shape != frame.get().shape:
                        overlay_frame = cv2.resize(overlay_frame, (frame.get().shape[1], frame.get().shape[0]))

                    masked_overlay = cv2.bitwise_and(overlay_frame, self.current_mask)
                    inverse_mask = cv2.bitwise_not(self.current_mask)
                    masked_frame = cv2.bitwise_and(frame, inverse_mask)
                    overlay_result = cv2.add(masked_frame, masked_overlay)

                    if self.frame_count < INITIAL_DELAY_FRAMES + TRANSITION_FRAMES:
                        self.transition_alpha = (self.frame_count - INITIAL_DELAY_FRAMES) / TRANSITION_FRAMES
                    else:
                        self.transition_alpha = 1.0

                    result = cv2.addWeighted(frame, 1.0 - self.transition_alpha, overlay_result, self.transition_alpha, 0.0)
                    result = result.get()

                # Add animated text after 3 seconds
                text_frame = create_animated_text_frame(FRAME_SIZE, TEXT, self.frame_count % 360, INITIAL_DELAY_FRAMES, self.blue_detected)
                text_mask = text_frame[:, :, 3] > 0
                result[text_mask] = text_frame[:, :, :3][text_mask]

            # Always show notifications if active
            if self.notification_frame_count < 60:
                notification_frame = create_notification_frame(FRAME_SIZE, self.notification_text, self.notification_frame_count)
                notification_mask = notification_frame[:, :, 3] > 0
                result[notification_mask] = notification_frame[:, :, :3][notification_mask]
                self.notification_frame_count += 1

            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            bytes_per_line = ch * w
            image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
        except Exception as e:
            self.notification_text = f"خطا: {str(e)}"
            self.notification_frame_count = 0

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.overlay_video:
            self.overlay_video.release()
        self.media_player.stop()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())