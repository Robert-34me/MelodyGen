import sys
import os
import random
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QListWidget, QLabel, QSplitter, QHBoxLayout, QMessageBox,
    QSlider, QListWidgetItem
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal
from audiocraft.models import musicgen
import torchaudio
import torch
from googletrans import Translator


class GenerateThread(QThread):
    finished = pyqtSignal(str, float)

    def __init__(self, model, prompt, track_folder):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.track_folder = track_folder

    def run(self):
        print("🎶 Началась генерация музыки...")
        start_time = time.time()
        wav = self.model.generate([self.prompt])
        track_num = len(os.listdir(self.track_folder)) + 1
        filename = f"track_{track_num}.wav"
        path = os.path.join(self.track_folder, filename)
        torchaudio.save(path, wav[0].cpu(), 32000)
        duration = time.time() - start_time
        print(f"✅ Генерация завершена за {duration:.2f} сек.")
        self.finished.emit(filename, duration)


class MusicApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MusicGen UI")
        self.setGeometry(100, 100, 900, 600)
        self.setWindowIcon(QIcon("Diplom(v3)/icon.png"))
        self.setStyleSheet("background-color: #111; color: #eee;")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Устройство генерации: {self.device.upper()}")

        self.model = musicgen.MusicGen.get_pretrained('facebook/musicgen-medium')
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model = self.model.model.to(self.device)

        self.model.set_generation_params(duration=15)

        self.player = QMediaPlayer()
        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)

        self.track_folder = "tracks"
        self.prompts_file = "prompts.txt"
        self.tracks_file = "tracks.txt"

        self.translator = Translator()

        if not os.path.exists(self.track_folder):
            os.makedirs(self.track_folder)

        self.tag_sets = [
            ["piano", "violin", "bass", "drums", "slow", "fast", "happy", "sad"],
            ["epic", "cinematic", "ambient", "dark", "synthwave", "retro", "lofi", "trap"],
            ["energetic", "melancholic", "classical", "jazz", "funk", "dreamy", "spacey", "industrial"]
        ]
        self.current_tag_set = 0

        self.init_ui()
        self.load_prompts()
        self.load_tracks()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("\U0001F3B5 Music Generator")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ff66cc; margin-bottom: 10px;")
        layout.addWidget(title)

        input_layout = QHBoxLayout()

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Введите запрос (например, dark synthwave in style of Mr Kitty)")
        self.prompt_input.setStyleSheet("background-color: #222; color: white; padding: 8px; border-radius: 6px;")
        input_layout.addWidget(self.prompt_input)

        dice_btn = QPushButton("🎲")
        dice_btn.setFixedWidth(40)
        dice_btn.clicked.connect(self.insert_random_prompt)
        input_layout.addWidget(dice_btn)

        layout.addLayout(input_layout)

        self.tag_layout = QHBoxLayout()
        self.add_tags()

        refresh_btn = QPushButton("🔁")
        refresh_btn.setFixedWidth(40)
        refresh_btn.clicked.connect(self.refresh_tags)
        self.tag_layout.addWidget(refresh_btn)

        layout.addLayout(self.tag_layout)

        self.generate_button = QPushButton("\U0001F3BC Сгенерировать музыку")
        self.generate_button.setStyleSheet("background-color: #ff66cc; color: black; font-weight: bold; padding: 8px; border-radius: 6px;")
        self.generate_button.clicked.connect(self.generate_music)
        layout.addWidget(self.generate_button)

        splitter = QSplitter(Qt.Horizontal)

        self.music_list = QListWidget()
        self.music_list.setStyleSheet("background-color: #222; color: white; border: none;")
        self.music_list.itemClicked.connect(self.play_selected_track)
        splitter.addWidget(self.music_list)

        self.prompt_list = QListWidget()
        self.prompt_list.setStyleSheet("background-color: #222; color: #ccc; border: none;")
        self.prompt_list.itemClicked.connect(self.use_prompt_from_list)
        splitter.addWidget(self.prompt_list)

        splitter.setSizes([600, 300])
        layout.addWidget(splitter)

        control_layout = QHBoxLayout()

        self.play_button = QPushButton("▶️")
        self.play_button.clicked.connect(self.toggle_play_pause)
        control_layout.addWidget(self.play_button)

        self.prev_button = QPushButton("⏮")
        self.prev_button.clicked.connect(self.play_previous)
        control_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("⏭")
        self.next_button.clicked.connect(self.play_next)
        control_layout.addWidget(self.next_button)

        self.delete_button = QPushButton("🗑 Удалить")
        self.delete_button.clicked.connect(self.delete_track)
        control_layout.addWidget(self.delete_button)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.player.setVolume)
        control_layout.addWidget(QLabel("Громкость:"))
        control_layout.addWidget(self.volume_slider)

        layout.addLayout(control_layout)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        layout.addWidget(self.position_slider)

        self.setLayout(layout)

    def add_tags(self):
        for btn in self.tag_layout.findChildren(QPushButton):
            btn.deleteLater()

        for tag in self.tag_sets[self.current_tag_set]:
            btn = QPushButton(tag)
            btn.setStyleSheet("background-color: #444; color: white; padding: 4px; border-radius: 4px;")
            btn.clicked.connect(lambda _, t=tag: self.add_tag(t))
            self.tag_layout.addWidget(btn)

    def refresh_tags(self):
        self.current_tag_set = (self.current_tag_set + 1) % len(self.tag_sets)
        self.add_tags()

    def insert_random_prompt(self):
        if self.prompt_list.count() > 0:
            random_prompt = self.prompt_list.item(random.randint(0, self.prompt_list.count() - 1)).text()
            self.prompt_input.setText(random_prompt)

    def add_tag(self, tag):
        current = self.prompt_input.text()
        if current and not current.endswith(","):
            current += ", "
        self.prompt_input.setText(current + tag)

    def generate_music(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return

        translated_prompt = self.translator.translate(prompt, src='ru', dest='en').text

        self.generate_button.setText("🎶 Генерация...")
        self.generate_button.setEnabled(False)

        self.thread = GenerateThread(self.model, translated_prompt, self.track_folder)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.start()

    def on_generation_finished(self, filename, duration):
        self.music_list.addItem(filename)
        with open(self.tracks_file, "a", encoding="utf-8") as f:
            f.write(filename + "\n")

        prompt = self.prompt_input.text().strip()
        if prompt not in [self.prompt_list.item(i).text() for i in range(self.prompt_list.count())]:
            self.prompt_list.addItem(prompt)
            with open(self.prompts_file, "a", encoding="utf-8") as f:
                f.write(prompt + "\n")

        self.generate_button.setText("\U0001F3BC Сгенерировать музыку")
        self.generate_button.setEnabled(True)

    def play_selected_track(self, item):
        self.play_track(item.text())

    def play_track(self, filename):
        path = os.path.join(self.track_folder, filename)
        url = QUrl.fromLocalFile(os.path.abspath(path))
        self.player.setMedia(QMediaContent(url))
        self.player.play()
        self.play_button.setText("⏸")
        self.current_track = filename

    def toggle_play_pause(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_button.setText("▶️")
        else:
            self.player.play()
            self.play_button.setText("⏸")

    def play_next(self):
        current_row = self.music_list.currentRow()
        if current_row < self.music_list.count() - 1:
            self.music_list.setCurrentRow(current_row + 1)
            self.play_selected_track(self.music_list.currentItem())

    def play_previous(self):
        current_row = self.music_list.currentRow()
        if current_row > 0:
            self.music_list.setCurrentRow(current_row - 1)
            self.play_selected_track(self.music_list.currentItem())

    def delete_track(self):
        current_item = self.music_list.currentItem()
        if not current_item:
            return

        reply = QMessageBox.question(self, "Удаление", f"Удалить {current_item.text()}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            os.remove(os.path.join(self.track_folder, current_item.text()))
            self.music_list.takeItem(self.music_list.row(current_item))

    def use_prompt_from_list(self, item):
        self.prompt_input.setText(item.text())

    def load_prompts(self):
        if not os.path.exists(self.prompts_file):
            with open(self.prompts_file, "w", encoding="utf-8") as f:
                f.write("dark synthwave beat in the style of Mr Kitty\n")

        with open(self.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.prompt_list.addItem(line)

    def load_tracks(self):
        if not os.path.exists(self.tracks_file):
            open(self.tracks_file, "w", encoding="utf-8").close()

        with open(self.tracks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(os.path.join(self.track_folder, line)):
                    self.music_list.addItem(line)

    def update_duration(self, duration):
        self.position_slider.setRange(0, duration)

    def update_position(self, position):
        self.position_slider.setValue(position)

    def set_position(self, position):
        self.player.setPosition(position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicApp()
    window.show()
    sys.exit(app.exec_())
