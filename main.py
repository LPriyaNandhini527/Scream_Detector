import os
import sqlite3
import threading
import logging
import re
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import bcrypt
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.menu import MDDropdownMenu
from kivy.properties import ObjectProperty
from alert_handler import send_alert_sms
from kivy.core.window import Window
Window.size = (360, 640)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KV = '''
ScreenManager:
    WelcomeScreen:
    SignInScreen:
    SignUpScreen:
    DetectionScreen:

<WelcomeScreen@Screen>:
    name: 'welcome'
    MDFloatLayout:
        canvas.before:
            Color:
                rgba: 0.1, 0, 0, 1
            Rectangle:
                pos: self.pos
                size: self.size

        MDLabel:
            text: "SCREAM DETECTION"
            font_style: "H3"
            halign: "center"
            pos_hint: {"center_y": 0.75}
            theme_text_color: "Custom"
            text_color: 1, 0.2, 0.2, 1

        MDLabel:
            text: "Protect. Prevent. Alert."
            font_style: "Subtitle1"
            halign: "center"
            pos_hint: {"center_y": 0.58}
            text_color: 1, 0.5, 0.3, 1

        MDRaisedButton:
            text: "SIGN IN"
            md_bg_color: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            size_hint_x: 0.6
            elevation: 10
            on_release: app.root.current = 'signin'

        MDRaisedButton:
            text: "SIGN UP"
            md_bg_color: 1, 0.4, 0, 1
            text_color: 1, 1, 1, 1
            pos_hint: {"center_x": 0.5, "center_y": 0.4}
            size_hint_x: 0.6
            elevation: 10
            on_release: app.root.current = 'signup'

<SignInScreen@Screen>:
    name: 'signin'
    MDFloatLayout:
        canvas.before:
            Color:
                rgba: 0.15, 0, 0, 1
            Rectangle:
                pos: self.pos
                size: self.size

        MDLabel:
            text: "Login"
            font_style: "H4"
            halign: "center"
            pos_hint: {"center_y": 0.88}
            theme_text_color: "Custom"
            text_color: 1, 0.3, 0.3, 1

        MDTextField:
            id: signin_email
            hint_text: "Email"
            icon_right: "email"
            pos_hint: {"center_x": 0.5, "center_y": 0.7}
            size_hint_x: 0.8
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1
            helper_text_mode: "on_focus"
            helper_text: "Enter your email"

        MDTextField:
            id: signin_password
            hint_text: "Password"
            icon_right: "lock"
            password: True
            pos_hint: {"center_x": 0.5, "center_y": 0.6}
            size_hint_x: 0.8
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1
            helper_text_mode: "on_focus"
            helper_text: "Enter your password"

        MDRaisedButton:
            id: signin_button
            text: "SIGN IN"
            md_bg_color: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1
            size_hint_x: 0.7
            pos_hint: {"center_x": 0.5, "center_y": 0.45}
            elevation: 10
            on_release: app.login_user()

        MDTextButton:
            text: "Don't have an account? Sign Up"
            pos_hint: {"center_x": 0.5, "center_y": 0.35}
            text_color: 1, 0.5, 0.3, 1
            on_release: app.root.current = 'signup'

<SignUpScreen@Screen>:
    name: 'signup'
    MDFloatLayout:
        canvas.before:
            Color:
                rgba: 0.15, 0, 0, 1
            Rectangle:
                pos: self.pos
                size: self.size

        MDLabel:
            text: "Register"
            font_style: "H4"
            halign: "center"
            pos_hint: {"center_y": 0.88}
            theme_text_color: "Custom"
            text_color: 1, 0.3, 0.3, 1

        MDTextField:
            id: full_name
            hint_text: "Full Name"
            icon_right: "account"
            pos_hint: {"center_x": 0.5, "center_y": 0.75}
            size_hint_x: 0.85
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1

        MDTextField:
            id: email
            hint_text: "Email"
            icon_right: "email"
            pos_hint: {"center_x": 0.5, "center_y": 0.68}
            size_hint_x: 0.85
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1

        MDTextField:
            id: gender
            hint_text: "Select Gender"
            readonly: True
            icon_right: "gender-male-female"
            pos_hint: {"center_x": 0.5, "center_y": 0.61}
            size_hint_x: 0.85
            on_focus: app.open_gender_menu(self) if self.focus else None
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1

        MDTextField:
            id: phone
            hint_text: "Phone Number"
            icon_right: "phone"
            input_filter: "int"
            pos_hint: {"center_x": 0.5, "center_y": 0.54}
            size_hint_x: 0.85
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1

        MDTextField:
            id: emergency
            hint_text: "Emergency Contact"
            icon_right: "account-alert"
            text: "+91"
            pos_hint: {"center_x": 0.5, "center_y": 0.47}
            size_hint_x: 0.85
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1

        MDTextField:
            id: password
            hint_text: "Password"
            icon_right: "eye-off"
            password: True
            pos_hint: {"center_x": 0.5, "center_y": 0.4}
            size_hint_x: 0.85
            mode: "rectangle"
            line_color_focus: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1
            on_icon_right: app.toggle_password_visibility(self)

        MDRaisedButton:
            id: signup_button
            text: "SIGN UP & START"
            md_bg_color: 1, 0.4, 0, 1
            text_color: 1, 1, 1, 1
            size_hint_x: 0.9
            pos_hint: {"center_x": 0.5, "center_y": 0.28}
            elevation: 10
            on_release: app.start_detection()

        MDTextButton:
            text: "Already have an account? Login"
            pos_hint: {"center_x": 0.5, "center_y": 0.2}
            text_color: 1, 0.5, 0.3, 1
            on_release: app.root.current = 'signin'

<DetectionScreen@Screen>:
    name: 'detection'
    MDFloatLayout:
        canvas.before:
            Color:
                rgba: 0.1, 0, 0, 1
            Rectangle:
                pos: self.pos
                size: self.size

        MDLabel:
            id: detection_msg
            text: "Detection Started"
            font_style: "H4"
            halign: "center"
            pos_hint: {"center_y": 0.6}
            theme_text_color: "Custom"
            text_color: 1, 0.3, 0.3, 1

        MDRaisedButton:
            text: "LOGOUT"
            md_bg_color: 1, 0.2, 0.2, 1
            text_color: 1, 1, 1, 1
            pos_hint: {"center_x": 0.5, "center_y": 0.4}
            elevation: 10
            on_release: app.confirm_logout()
'''

class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 40, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 10 * 4, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class WelcomeScreen(Screen): pass
class SignInScreen(Screen): pass
class SignUpScreen(Screen): pass
class DetectionScreen(Screen): pass

class ScreamDetectionApp(MDApp):
    gender_menu = ObjectProperty(None)
    dialog = None
    models_loaded = False
    detection_thread = None
    detection_stop_event = threading.Event()
    db_lock = threading.Lock()
    alert_cooldown = False
    detection_sensitivity = 0.5  # default threshold for detection

    def build(self):
        self.init_db()
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.theme_style = "Dark"
        return Builder.load_string(KV)

    def toggle_password_visibility(self, password_field):
        password_field.password = not password_field.password
        if password_field.password:
            password_field.icon_right = "eye-off"
        else:
            password_field.icon_right = "eye"

    def init_db(self):
        import db
        db.init_db()

    def show_dialog(self, title, message):
        if self.dialog:
            self.dialog.dismiss()
        self.dialog = MDDialog(
            title=title,
            text=message,
            buttons=[
                MDRaisedButton(
                    text="OK", on_release=lambda x: self.dialog.dismiss()
                )
            ],
        )
        self.dialog.open()

    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

    def validate_password_strength(self, password):
        # At least 8 chars, one uppercase, one lowercase, one digit
        pattern = r'^(?=.[a-z])(?=.[A-Z])(?=.*\d).{8,}$'
        return re.match(pattern, password) is not None

    def validate_and_format_phone(self, phone):
        phone = phone.strip()
        if not phone.startswith("+"):
            phone = "+91" + phone
        return phone

    def login_user(self):
        try:
            screen = self.root.get_screen('signin')
            email = screen.ids.signin_email.text.strip()
            password = screen.ids.signin_password.text.strip()

            if not email or not password:
                self.show_dialog("Missing Info", "Please enter both email and password.")
                return

            if not self.validate_email(email):
                self.show_dialog("Invalid Email", "Please enter a valid email address.")
                return

            import db
            with self.db_lock:
                user = db.get_user_by_email(email)
            if not user:
                self.show_dialog("Login Failed", "Invalid email or password.")
                return

            stored_hash = user['password']
            logging.info(f"Stored hash type: {type(stored_hash)}, value (partial): {str(stored_hash)[:30]}")

            if isinstance(stored_hash, str):
                try:
                    stored_hash = stored_hash.encode('utf-8')
                except Exception as e:
                    logging.error(f"Error encoding stored_hash: {e}", exc_info=True)
                    self.show_dialog("Login Failed", "Invalid email or password.")
                    return
            try:
                # Check if stored_hash is a valid bcrypt hash
                if not (stored_hash.startswith(b"$2a$") or stored_hash.startswith(b"$2b$") or stored_hash.startswith(b"$2y$")):
                    logging.error("Stored password hash is not a valid bcrypt hash.")
                    self.show_dialog("Login Failed", "Invalid email or password.")
                    return
                if not bcrypt.checkpw(password.encode(), stored_hash):
                    self.show_dialog("Login Failed", "Invalid email or password.")
                    return
            except ValueError as ve:
                logging.error(f"Invalid salt error in bcrypt: {ve}", exc_info=True)
                self.show_dialog("Login Failed", "Invalid email or password.")
                return

            full_name = user['full_name']
            self.root.get_screen('detection').ids.detection_msg.text = f"👋 Welcome {full_name}, detection started!"
            self.root.current = 'detection'
            self.load_models()
            self.start_real_time_detection(email)
        except Exception as e:
            logging.error(f"Error in login_user: {e}", exc_info=True)
            self.show_dialog("Error", "An error occurred during login. Please try again.")

    def start_detection(self):
        try:
            signup = self.root.get_screen('signup')
            full_name = signup.ids.full_name.text.strip()
            email = signup.ids.email.text.strip()
            gender = signup.ids.gender.text.strip()
            phone = signup.ids.phone.text.strip()
            emergency = signup.ids.emergency.text.strip()
            password = signup.ids.password.text.strip()

            if not all([full_name, email, gender, phone, emergency, password]):
                self.show_dialog("Incomplete Form", "Please fill all the fields.")
                return

            if not self.validate_email(email):
                self.show_dialog("Invalid Email", "Please enter a valid email address.")
                return

            # Removed password strength validation to accept any password
            # if not self.validate_password_strength(password):
            #     self.show_dialog("Weak Password", "Password must be at least 8 characters long, include uppercase, lowercase, and a digit.")
            #     return

            emergency = self.validate_and_format_phone(emergency)
            phone = self.validate_and_format_phone(phone)

            import db
            with self.db_lock:
                if db.get_user_by_email(email):
                    self.show_dialog("Email Exists", "An account with this email already exists.")
                    return
                db.register_user(full_name, email, gender, phone, emergency, bcrypt.hashpw(password.encode(), bcrypt.gensalt()))

            self.root.get_screen('detection').ids.detection_msg.text = f"Scream Detection Started for {full_name}!"
            self.root.current = 'detection'
            self.load_models()
            self.start_real_time_detection(email)
        except Exception as e:
            logging.error(f"Error in start_detection: {e}", exc_info=True)
            self.show_dialog("Error", "An error occurred during signup. Please try again.")

    def open_gender_menu(self, caller):
        gender_options = ["Male", "Female", "Other"]
        menu_items = [{"text": g, "on_release": lambda x=g: self.set_gender(x, caller)} for g in gender_options]
        self.gender_menu = MDDropdownMenu(caller=caller, items=menu_items, width_mult=3)
        self.gender_menu.open()

    def set_gender(self, gender, caller):
        caller.text = gender
        if self.gender_menu:
            self.gender_menu.dismiss()

    def load_models(self):
        if not self.models_loaded:
            try:
                self.cnn1d_model = CNN1DModel()
                self.cnn1d_model.load_state_dict(torch.load("scream_detector_cnn1d.pth"))
                self.cnn1d_model.eval()

                self.cnn2d_model = CNN2DModel()
                self.cnn2d_model.load_state_dict(torch.load("scream_detector_cnn2d.pth"))
                self.cnn2d_model.eval()

                self.lstm_model = LSTMModel()
                self.lstm_model.load_state_dict(torch.load("scream_detector_lstm.pth"))
                self.lstm_model.eval()

                self.models_loaded = True
                logging.info("PyTorch models loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading models: {e}", exc_info=True)
                self.show_dialog("Model Load Error", f"Failed to load models: {e}")

    def get_gps_location(self):
        try:
            response = requests.get('https://ipinfo.io/json')
            data = response.json()
            city = data.get('city', '')
            region = data.get('region', '')
            country = data.get('country', '')
            location_name = ', '.join(filter(None, [city, region, country]))
            return location_name if location_name else None
        except Exception as e:
            logging.error(f"Failed to get GPS location: {e}", exc_info=True)
            return None

    def start_real_time_detection(self, user_email):
        if self.detection_thread and self.detection_thread.is_alive():
            logging.info("Detection thread already running.")
            return

        # Ensure models are loaded before starting detection
        if not self.models_loaded:
            self.load_models()

        self.detection_stop_event.clear()

        cnn1d_model = self.cnn1d_model
        cnn2d_model = self.cnn2d_model
        lstm_model = self.lstm_model
        detection_sensitivity = self.detection_sensitivity
        alert_cooldown_flag = [self.alert_cooldown]  # use list to allow modification in nested function
        alert_cooldown_lock = threading.Lock()

        def reset_alert_cooldown_flag():
            with alert_cooldown_lock:
                alert_cooldown_flag[0] = False

        def detect():
            duration = 3  # seconds
            sample_rate = 22050

            try:
                import db
                with self.db_lock:
                    emergency_contact = db.get_emergency_contact_by_email(user_email)

                if not emergency_contact:
                    self.show_dialog("Error", "No emergency contact found for user.")
                    return

                logging.info("Real-time scream detection started.")
                while not self.detection_stop_event.is_set():
                    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    audio = audio.flatten()
                    features = self.extract_realtime_features(audio, sr=sample_rate)

                    input_1d = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()
                    input_2d = torch.tensor(features.reshape(10, 4)).unsqueeze(0).unsqueeze(0).float()
                    input_lstm = torch.tensor(features).unsqueeze(0).unsqueeze(1).float()

                    with torch.no_grad():
                        pred1 = cnn1d_model(input_1d)
                        pred2 = cnn2d_model(input_2d)
                        pred3 = lstm_model(input_lstm)

                        avg_pred = (pred1 + pred2 + pred3) / 3
                        final_label = torch.argmax(avg_pred, dim=1).item()
                        logging.info(f"Predictions: CNN1D={pred1}, CNN2D={pred2}, LSTM={pred3}, Avg={avg_pred}, Final Label={final_label}")

                    if final_label == 1 and avg_pred[0][1].item() >= detection_sensitivity:
                        logging.info("Scream detected!")
                        self.root.get_screen('detection').ids.detection_msg.text = " Scream detected! Sending alert..."
                        location = self.get_gps_location()
                        message = "Scream detected! Please check immediately."
                        if location:
                            message += f" Location: {location}"
                        with alert_cooldown_lock:
                            if not alert_cooldown_flag[0]:
                                send_alert_sms(message, emergency_contact)
                                alert_cooldown_flag[0] = True
                                threading.Timer(60, reset_alert_cooldown_flag).start()
                    else:
                        self.root.get_screen('detection').ids.detection_msg.text = "No scream detected."

            except Exception as e:
                logging.error(f"Detection error: {e}", exc_info=True)

        self.detection_thread = threading.Thread(target=detect, daemon=True)
        self.detection_thread.start()

    def reset_alert_cooldown(self):
        self.alert_cooldown = False

    def extract_realtime_features(self, audio, sr=22050):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

    def confirm_logout(self):
        if self.dialog:
            self.dialog.dismiss()
        self.dialog = MDDialog(
            title="Confirm Logout",
            text="Are you sure you want to logout?",
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: self.dialog.dismiss()),
                MDRaisedButton(text="Logout", on_release=self.logout)
            ],
        )
        self.dialog.open()

    def logout(self, *args):
        self.detection_stop_event.set()
        self.models_loaded = False
        self.root.current = 'welcome'
        self.show_dialog("Logged Out", "You have been logged out successfully.")

if __name__ == '__main__':
    ScreamDetectionApp().run()