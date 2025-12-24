import cv2
import os
from datetime import datetime

def get_camera_resolutions():
    """Get available resolutions for the camera."""
    cap = cv2.VideoCapture(0)
    resolutions = [
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
    ]
    
    available = []
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            available.append((w, h))
    
    cap.release()
    return available

def set_camera_focus(cap, focus_value):
    """Set camera focus. focus_value: 0-255 (0=auto, 1-255=manual)"""
    # Disable autofocus first
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # Set manual focus
    cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    print(f"Focus set to: {focus_value}")

def main():
    # Create output folder if it doesn't exist
    output_folder = "saved_video"
    os.makedirs(output_folder, exist_ok=True)
    
    print("=== USB Camera Video Recorder ===\n")
    
    # Get available resolutions
    available_res = get_camera_resolutions()
    if not available_res:
        print("Error: No camera found or no resolutions available!")
        return
    
    print("Available resolutions:")
    for i, (w, h) in enumerate(available_res, 1):
        print(f"{i}. {w}x{h}")
    
    # User selects resolution
    while True:
        try:
            choice = int(input("\nSelect resolution (enter number): ") or "3")
            if 1 <= choice <= len(available_res):
                width, height = available_res[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_res)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\nSelected resolution: {width}x{height}")
    
    # Get FPS from user
    while True:
        try:
            fps = int(input("Enter FPS (frames per second, default 25): ") or "25")
            if fps > 0:
                break
            else:
                print("FPS must be greater than 0")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get duration from user
    while True:
        try:
            duration = int(input("Enter recording duration in seconds (default 10): ") or "10")
            if duration > 0:
                break
            else:
                print("Duration must be greater than 0")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return
    
    # Camera focus and exposure settings
    print("\n=== Camera Settings ===")
    
    # Auto-focus (set to 0 for manual, 1 for auto)
    autofocus_choice = input("Enable autofocus? (y/n, default y): ").lower().strip() or "y"
    if autofocus_choice == 'y':
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print("Autofocus: ENABLED")
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Manual focus (0-255, higher = farther focus distance)
        while True:
            try:
                focus = int(input("Enter manual focus value (0-255, default 50): ") or "50")
                if 0 <= focus <= 255:
                    set_camera_focus(cap, focus)
                    break
                else:
                    print("Focus value must be between 0 and 255")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Exposure compensation
    exposure_choice = input("Adjust exposure? (y/n, default n): ").lower().strip() or "n"
    if exposure_choice == 'y':
        while True:
            try:
                exposure = int(input("Enter exposure compensation (-8 to 8, default 0): ") or "0")
                if -8 <= exposure <= 8:
                    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                    print(f"Exposure set to: {exposure}")
                    break
                else:
                    print("Exposure must be between -8 and 8")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Brightness
    brightness_choice = input("Adjust brightness? (y/n, default n): ").lower().strip() or "n"
    if brightness_choice == 'y':
        while True:
            try:
                brightness = int(input("Enter brightness (0-100, default 50): ") or "50")
                if 0 <= brightness <= 100:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness / 100.0)
                    print(f"Brightness set to: {brightness}")
                    break
                else:
                    print("Brightness must be between 0 and 100")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Contrast
    contrast_choice = input("Adjust contrast? (y/n, default n): ").lower().strip() or "n"
    if contrast_choice == 'y':
        while True:
            try:
                contrast = int(input("Enter contrast (0-200, default 50): ") or "50")
                if 0 <= contrast <= 100:
                    cap.set(cv2.CAP_PROP_CONTRAST, contrast / 100.0)
                    print(f"Contrast set to: {contrast}")
                    break
                else:
                    print("Contrast must be between 0 and 100")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"video_{timestamp}_{width}x{height}_{fps}fps.mp4")
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Cannot create video writer!")
        cap.release()
        return
    
    print(f"\nRecording to: {output_path}")
    print(f"Duration: {duration} seconds")
    print("Press 'q' to stop recording early.\n")
    
    frame_count = 0
    total_frames = duration * fps
    
    try:
        while frame_count < total_frames:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera!")
                break
            
            # Write frame to video
            out.write(frame)
            frame_count += 1
            
            # Display progress
            progress = (frame_count / total_frames) * 100
            print(f"\rRecording... {frame_count}/{total_frames} frames ({progress:.1f}%)", end='', flush=True)
            
            # Optional: display live preview
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user.")
    
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n\nVideo saved successfully!")
        print(f"Output: {output_path}")
        print(f"Total frames recorded: {frame_count}")

if __name__ == '__main__':
    main()