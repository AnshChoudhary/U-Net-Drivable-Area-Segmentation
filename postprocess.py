import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os

def morphological_cleanup(frame, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def temporal_smoothing(frames, alpha=0.8):
    result = frames[0].copy().astype(float)
    for i in range(1, len(frames)):
        result = alpha * result + (1 - alpha) * frames[i].astype(float)
    return result.astype(np.uint8)

def optical_flow_stabilization(prev, curr):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = prev.shape[:2]
    y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(np.float32)
    pos = np.vstack((x, y)).T
    
    shifted = pos + flow.reshape(-1, 2)
    shifted = shifted.reshape(h, w, 2)
    
    warped = cv2.remap(prev, shifted, None, cv2.INTER_LINEAR)
    
    return cv2.addWeighted(warped, 0.5, curr, 0.5, 0)

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file '{output_path}'.")
        cap.release()
        return

    frame_buffer = []
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply morphological cleanup
        cleaned = morphological_cleanup(frame)
        
        # Apply optical flow stabilization if we have a previous frame
        if prev_frame is not None:
            stabilized = optical_flow_stabilization(prev_frame, cleaned)
        else:
            stabilized = cleaned
        
        # Add to frame buffer
        frame_buffer.append(stabilized)
        if len(frame_buffer) > 7:  # Keep a buffer of 7 frames
            frame_buffer.pop(0)
        
        # Apply temporal smoothing
        smoothed = temporal_smoothing(frame_buffer)
        
        # Final gaussian blur for extra smoothness
        final = gaussian_filter(smoothed, sigma=1)
        
        # Write the frame
        out.write(final.astype(np.uint8))
        
        prev_frame = cleaned
        frame_count += 1
        
        # Display progress
        print(f"Processed frame {frame_count}", end='\r')
    
    # Release everything
    cap.release()
    out.release()
    
    if frame_count > 0:
        print(f"\nProcessing complete! Output saved to {output_path}")
    else:
        print("\nError: No frames were processed. The input video might be empty.")

# Usage
input_video = '/Users/anshchoudhary/Downloads/u-net-torch/output_video_masks_only.mp4'
output_video = '/Users/anshchoudhary/Downloads/u-net-torch/postprocessed_mask_video.mp4'
process_video(input_video, output_video)