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
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'MJPG', 'XVID', 'DIVX'
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
            
            # Optional: display live preview (uncomment to see)
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