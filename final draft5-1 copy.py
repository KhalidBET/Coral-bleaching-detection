import customtkinter
from PIL import Image, ImageTk
from picamera2 import Picamera2
import time
from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import os
import gc  # Import garbage collector
import webbrowser

web_path = '/home/rapi/Desktop/html wewb/updated-coral-html.html' 

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="4BDSWSo79P86BTJjT3dn"
)

# Initialize CustomTkinter
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("CORAL CLASSICATION")
        self.geometry("920x500]")

        # Create main frame
        self.main_frame = customtkinter.CTkFrame(master=self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create camera frame
        self.camera_frame = customtkinter.CTkFrame(master=self.main_frame)
        self.camera_frame.grid(row=0, column=1, sticky="ne")

        self.camera_canvas = customtkinter.CTkCanvas(master=self.camera_frame, width=640, height=480)
        self.camera_canvas.grid(row=0, column=0, sticky="nsew")

        # Create image frame
        self.image_frame = customtkinter.CTkFrame(master=self.main_frame)
        self.image_frame.grid(row=0, column=0, sticky="nw")

        # Load and display an image
        try:
            self.img_path = "/home/rapi/Downloads/logo.jpeg"
            image = Image.open(self.img_path)
            image = image.resize((200, 200))
            ctk_image = customtkinter.CTkImage(light_image=image, dark_image=image, size=(200, 200))
            self.image_label = customtkinter.CTkLabel(master=self.image_frame, image=ctk_image, text="")
            self.image_label.pack(padx=30, pady=0)
        except FileNotFoundError:
            print(f"Image not found at: {self.img_path}")

        # Create button frame
        self.button_frame = customtkinter.CTkFrame(master=self.main_frame)
        self.button_frame.grid(row=0, column=0, sticky="nw", padx=20, pady=220)

        # Create buttons
        buttons = [
            ("Automatic", self.button1_clicked),
            ("Manual", self.button2_clicked),
            ("Process", self.button3_clicked),
            ("About us", self.button4_clicked),
            ("History", self.show_history)
        ]
        
        for text, command in buttons:
            btn = customtkinter.CTkButton(
                master=self.button_frame,
                text=text,
                command=command,
                width=220
            )
            btn.pack(pady=10)

        # --- Camera Resolution Settings ---
        self.width = 640
        self.height = 600

        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (self.width, self.height)})
        self.picam2.configure(config)
        self.picam2.start()

        # Initialize history variables
        self.history_images = []
        self.current_history_index = 0

        # Start camera feed
        self.update_frame()

    def update_frame(self):
        frame = self.picam2.capture_array()
        pil_image = Image.fromarray(frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        self.camera_canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.camera_canvas.image = tk_image
        self.after(20, self.update_frame)

    def button1_clicked(self):
        image_counter = 1
        self.is_automatic_running = True

        while self.is_automatic_running: 
            filename = f"ROV_{image_counter:04d}.jpg"
            image_path = os.path.join("/home/rapi/Desktop/coral detect/history", filename)
            print(image_path)
            
            frame = self.picam2.capture_array()
            time.sleep(5)
            
            cv2.imwrite(image_path, frame)
            result = CLIENT.infer(image_path, model_id="coral-bleaching-final/8")
            print(result)

            det = result['predictions']
            if len(det) == 0:
                print("no coral detected")

            image = cv2.imread(image_path)
            detections = sv.Detections.from_inference(result)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            
            annotated_image_path = os.path.join("/home/rapi/Desktop/coral detect/detecthistory", f"ROV_{image_counter:04d}_annotated.jpg")
            cv2.imwrite(annotated_image_path, annotated_image)
            sv.plot_image(image=annotated_image, size=(8, 8))
            
            image_counter += 1
            time.sleep(5)

    def button2_clicked(self):
        self.is_automatic_running = False 
        print("Manual mode activated. Automatic loop stopped.")
        
    def button3_clicked(self):
        # Get the current image number being displayed
        if hasattr(self, 'current_history_index') and self.history_images:
            # Get the filename of the current image
            history_folder = "/home/rapi/Desktop/coral detect/history"
            image_files = sorted([f for f in os.listdir(history_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if self.current_history_index < len(image_files):
                current_filename = image_files[self.current_history_index]
                
                # Extract the image number from the filename (SWAT_XXXX.jpg)
                image_number = current_filename.split('_')[1].split('.')[0]
                
                # Construct the path to the corresponding processed image
                processed_filename = f"ROV_{image_number}_annotated.jpg"
                processed_image_path = os.path.join("/home/rapi/Desktop/coral detect/detecthistory", processed_filename)
                
                if os.path.exists(processed_image_path):
                    try:
                        # Create a new window to display the processed image
                        processed_window = customtkinter.CTkToplevel(self)
                        processed_window.title(f"Processed Image - {processed_filename}")
                        processed_window.geometry("700x580")
                        
                        # Create a frame for the image
                        frame = customtkinter.CTkFrame(processed_window)
                        frame.pack(padx=10, pady=10, fill="both", expand=True)
                        
                        # Load and display the processed image
                        pil_image = Image.open(processed_image_path)
                        pil_image = pil_image.resize((640, 480))
                        ctk_image = customtkinter.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
                        
                        image_label = customtkinter.CTkLabel(
                            master=frame,
                            text="",
                            image=ctk_image
                        )
                        image_label.pack(padx=10, pady=10)
                        
                        # Keep a reference to prevent garbage collection
                        image_label.image = ctk_image
                        
                        # Add a close button
                        close_button = customtkinter.CTkButton(
                            master=frame,
                            text="Close",
                            command=processed_window.destroy,
                            width=120
                        )
                        close_button.pack(pady=10)
                        
                    except Exception as e:
                        print(f"Error loading processed image: {e}")
                else:
                    print(f"Processed image not found: {processed_image_path}")
        else:
            print("No image currently selected in history view")
            
    def button4_clicked(self):
        # Open the HTML file in the default web browser
        webbrowser.open_new_tab('file:///' + web_path) 

    def show_history(self):
        history_folder = "/home/rapi/Desktop/coral detect/history"
        print(f"History Folder: {history_folder}")

        self.history_images = []
        image_files = sorted([f for f in os.listdir(history_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(image_files)} images in the history folder.")
        
        self.camera_frame.grid_forget()
        
        for filename in image_files:
            filepath = os.path.join(history_folder, filename)
            print(f"Loading image: {filepath}")
            try:
                pil_image = Image.open(filepath)
                pil_image = pil_image.resize((640, 480))
                ctk_image = customtkinter.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
                self.history_images.append(ctk_image)
            except Exception as e:
                print(f"Error loading image: {filepath} - {e}")
        
        if self.history_images:
            self.current_history_index = 0
            
            if not hasattr(self, 'history_display_frame'):
                self.history_display_frame = customtkinter.CTkFrame(master=self.main_frame)
                self.history_display_frame.grid(row=0, column=1, sticky="nsew")
                
                self.history_image_label = customtkinter.CTkLabel(
                    master=self.history_display_frame,
                    text="",
                    image=None,
                    width=640,
                    height=480
                )
                self.history_image_label.grid(row=0, column=0, sticky="nsew")
            
            def prev_click():
                if self.current_history_index > 0:
                    self.current_history_index -= 1
                    self.display_history_image()
                    print(f"Previous: Showing image {self.current_history_index + 1} of {len(self.history_images)}")

            def next_click():
                if self.current_history_index < len(self.history_images) - 1:
                    self.current_history_index += 1
                    self.display_history_image()
                    print(f"Next: Showing image {self.current_history_index + 1} of {len(self.history_images)}")

            def live_feed_click():
                self.show_live_feed()
                print("Switching to live feed")
            
            for widget in self.button_frame.winfo_children():
                widget.destroy()
            
            buttons = [
                ("Automatic", self.button1_clicked),
                ("Manual", self.button2_clicked),
                ("Process", self.button3_clicked),
                ("Save", self.button4_clicked),
                ("Previous", prev_click),
                ("Next", next_click),
                ("Live Feed", live_feed_click)
            ]
            
            for text, command in buttons:
                btn = customtkinter.CTkButton(
                    master=self.button_frame,
                    text=text,
                    command=command,
                    width=220
                )
                btn.pack(pady=10)

            self.display_history_image()
        else:
            print("No images found in history folder")

    def display_history_image(self):
        if hasattr(self, 'history_image_label') and self.history_images:
            current_image = self.history_images[self.current_history_index]
            self.history_image_label.configure(image=current_image)
            print(f"Displaying image {self.current_history_index + 1} of {len(self.history_images)}")

    def show_live_feed(self):
        # Clear existing buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()
        
        # Recreate original buttons
        buttons = [
            ("Automatic", self.button1_clicked),
            ("Manual", self.button2_clicked),
            ("Process", self.button3_clicked),
            ("Save", self.button4_clicked),
            ("History", self.show_history)
        ]
        
        for text, command in buttons:
            btn = customtkinter.CTkButton(
                master=self.button_frame,
                text=text,
                command=command,
                width=220
            )
            btn.pack(pady=10)
        
        # Destroy history display frame if it exists
        if hasattr(self, 'history_display_frame'):
            self.history_display_frame.destroy()
            delattr(self, 'history_display_frame')
        
        # Recreate camera frame and canvas
        if hasattr(self, 'camera_frame'):
            self.camera_frame.destroy()
        
        self.camera_frame = customtkinter.CTkFrame(master=self.main_frame)
        self.camera_frame.grid(row=0, column=1, sticky="ne")
        
        self.camera_canvas = customtkinter.CTkCanvas(
            master=self.camera_frame,
            width=640,
            height=480
        )
        self.camera_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Clear history images and force garbage collection
        self.history_images = []
        gc.collect()
        
        # Make sure camera is running
        if not self.picam2.is_running():
            config = self.picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": (self.width, self.height)}
            )
            self.picam2.configure(config)
            self.picam2.start()
        
        # Restart the frame updates
        self.update_frame()
        print("Live feed restarted")

    def update_frame(self):
        try:
            if hasattr(self, 'camera_canvas') and self.camera_frame.winfo_exists():
                frame = self.picam2.capture_array()
                pil_image = Image.fromarray(frame)
                tk_image = ImageTk.PhotoImage(image=pil_image)
                self.camera_canvas.create_image(0, 0, anchor="nw", image=tk_image)
                self.camera_canvas.image = tk_image  # Keep a reference
                self.after(20, self.update_frame)
        except Exception as e:
            print(f"Error updating frame: {e}")
            self.after(20, self.update_frame)  # Keep trying to update

if __name__ == "__main__":
    app = App()
    app.mainloop()

