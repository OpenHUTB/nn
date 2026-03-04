import cv2
import numpy as np
import os
from scipy import stats

def grayscale(img):
    """Convert image to grayscale to reduce computation"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size=5):
    """Gaussian blur for noise reduction"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge(img, low_threshold=50, high_threshold=150):
    """Canny edge detection"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """Define region of interest to keep only lane-related area"""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    """Fit and draw left/right lane lines, return fitting parameters and bottom endpoints"""
    left_points = []   
    right_points = []  
    
    if lines is None:
        return img, None, None, None, None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            
            # Wider slope range for better detection
            if -1.2 < slope < -0.1:
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            elif 0.1 < slope < 1.2:
                right_points.append([x1, y1])
                right_points.append([x2, y2])
    
    # Remove outliers with Z-score (relaxed threshold)
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    if len(left_points) > 2:
        z_scores = stats.zscore(left_points)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)
        left_points = left_points[filtered_entries]
    
    if len(right_points) > 2:
        z_scores = stats.zscore(right_points)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)
        right_points = right_points[filtered_entries]
    
    height, width = img.shape[:2]
    y_bottom = height
    y_top = int(height * 0.5)  # Higher ROI top for more lane coverage
    
    left_fit = None
    right_fit = None
    left_bottom_x = None
    right_bottom_x = None
    
    if len(left_points) >= 2:
        left_y = left_points[:, 1]
        left_x = left_points[:, 0]
        left_fit = np.polyfit(left_y, left_x, 2)
        
        y_vals = np.array([y_bottom, y_top])
        x_vals = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
        
        x1_left = np.clip(int(x_vals[0]), 0, width)
        x2_left = np.clip(int(x_vals[1]), 0, width)
        
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
        left_bottom_x = x1_left
    
    if len(right_points) >= 2:
        right_y = right_points[:, 1]
        right_x = right_points[:, 0]
        right_fit = np.polyfit(right_y, right_x, 2)
        
        y_vals = np.array([y_bottom, y_top])
        x_vals = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
        
        x1_right = np.clip(int(x_vals[0]), 0, width)
        x2_right = np.clip(int(x_vals[1]), 0, width)
        
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
        right_bottom_x = x1_right
    
    return img, left_fit, right_fit, left_bottom_x, right_bottom_x

def calculate_car_offset(left_bottom_x, right_bottom_x, img_shape):
    """Calculate vehicle offset from lane center (meters)"""
    if left_bottom_x is None or right_bottom_x is None:
        return "N/A"
    
    height, width = img_shape[:2]
    
    lane_width_pix = abs(right_bottom_x - left_bottom_x)
    if lane_width_pix < 80:  # Relaxed minimum width threshold
        return "N/A"
    
    # Standard lane width 3.7 meters for pixel to meter conversion
    xm_per_pix = 3.7 / lane_width_pix
    
    lane_center_x = (left_bottom_x + right_bottom_x) / 2
    car_center_x = width / 2
    
    camera_offset_pix = 0
    car_center_x += camera_offset_pix
    
    offset_pix = car_center_x - lane_center_x
    offset_m = offset_pix * xm_per_pix
    
    if offset_m > 0.02:
        return f"Right {abs(offset_m):.3f}m"
    elif offset_m < -0.02:
        return f"Left {abs(offset_m):.3f}m"
    else:
        return "Centered (±0.02m)"

def calculate_lane_width(left_bottom_x, right_bottom_x, left_fit, right_fit, img_shape):
    """
    Optimized lane width calculation:
    1. More sampling points
    2. Wider valid range
    3. Fallback strategy for single point calculation
    4. User-friendly error messages
    """
    if left_bottom_x is None or right_bottom_x is None:
        return "Lane lines not detected"
    if left_fit is None or right_fit is None:
        return "Lane line fitting failed"
    
    height, width = img_shape[:2]
    
    # More sampling points (15 instead of 10)
    sample_heights = np.linspace(height * 0.5, height, num=15)
    lane_widths_pix = []
    
    for y in sample_heights:
        left_x = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        right_x = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        
        if 0 <= left_x <= width and 0 <= right_x <= width:
            lane_width_pix = abs(right_x - left_x)
            # Wider valid pixel range (40-1200 instead of 50-1000)
            if 40 < lane_width_pix < 1200:
                lane_widths_pix.append(lane_width_pix)
    
    # Fallback strategy - calculate with bottom point if no samples
    if len(lane_widths_pix) == 0:
        y = height
        left_x = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        right_x = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        if 0 <= left_x <= width and 0 <= right_x <= width:
            lane_width_pix = abs(right_x - left_x)
            if 40 < lane_width_pix < 1200:
                lane_widths_pix.append(lane_width_pix)
    
    if len(lane_widths_pix) == 0:
        return "Width calculation failed"
    
    avg_width_pix = np.mean(lane_widths_pix)
    
    xm_per_pix = 3.7 / 700
    avg_width_m = avg_width_pix * xm_per_pix
    
    # Wider valid meter range (2.0-5.0 instead of 2.5-4.5)
    if 2.0 <= avg_width_m <= 5.0:
        return f"{avg_width_m:.2f} m"
    else:
        return f"Abnormal width ({avg_width_m:.2f} m)"

def calculate_lane_curvature(left_fit, right_fit, img_shape):
    """Calculate lane curvature radius (meters)"""
    if left_fit is None or right_fit is None:
        return "N/A"
    
    height, width = img_shape[:2]
    
    ym_per_pix = 30 / 720  
    xm_per_pix = 3.7 / 700  
    
    y_vals = np.linspace(0, height-1, num=50)
    y_eval = np.max(y_vals)
    
    left_x = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
    left_fit_cr = np.polyfit(y_vals * ym_per_pix, left_x * xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    
    right_x = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
    right_fit_cr = np.polyfit(y_vals * ym_per_pix, right_x * xm_per_pix, 2)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[1])
    
    avg_curvature = (left_curverad + right_curverad) / 2
    
    if avg_curvature < 100 or avg_curvature > 10000:
        return "N/A"
    
    if avg_curvature > 1000:
        return f"{int(avg_curvature)} m (Straight)"
    else:
        return f"{avg_curvature:.1f} m"

def lane_detection_pipeline(img):
    """Complete lane detection pipeline (optimized version)"""
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny_edge(blur)
    
    height, width = img.shape[:2]
    vertices = np.array([[
        (width*0.05, height),
        (width*0.4, height*0.5),
        (width*0.6, height*0.5),
        (width*0.95, height)
    ]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)
    
    # Optimized Hough transform parameters
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,               
        theta=np.pi/180,     
        threshold=25,        
        minLineLength=40,    
        maxLineGap=20        
    )
    
    line_img = np.zeros_like(img)
    line_img, left_fit, right_fit, left_bottom_x, right_bottom_x = draw_lines(line_img, lines)
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    # Draw curvature
    curvature = calculate_lane_curvature(left_fit, right_fit, img.shape)
    cv2.putText(
        result, 
        f"Lane Curvature: {curvature}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # Draw offset
    offset = calculate_car_offset(left_bottom_x, right_bottom_x, img.shape)
    cv2.putText(
        result, 
        f"Car Offset: {offset}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # Draw width with color coding
    lane_width = calculate_lane_width(left_bottom_x, right_bottom_x, left_fit, right_fit, img.shape)
    if "m" in lane_width and "Abnormal" not in lane_width:
        width_color = (0, 255, 255)  # Normal: Yellow
    elif "Abnormal" in lane_width:
        width_color = (0, 165, 255)  # Abnormal: Orange
    else:
        width_color = (0, 0, 255)    # Failed: Red
    
    cv2.putText(
        result, 
        f"Lane Width: {lane_width}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        width_color,
        2
    )
    
    return result

def batch_detect_images(folder_path):
    """Batch detect all images in specified folder"""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder does not exist → {folder_path}")
        return
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if len(img_files) == 0:
        print(f"❌ Error: No supported images found in folder → {folder_path}")
        return
    
    print(f"\n📁 Starting batch detection, found {len(img_files)} images...")
    success_count = 0
    
    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"\n[{idx}/{len(img_files)}] Processing: {img_file}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Skipped: Cannot read image {img_file}")
            continue
        
        result = lane_detection_pipeline(img)
        
        save_name = os.path.splitext(img_file)[0] + "_batch_result.jpg"
        save_path = os.path.join(folder_path, save_name)
        cv2.imwrite(save_path, result)
        
        success_count += 1
        print(f"✅ Completed: Result saved as {save_name}")
    
    print(f"\n🎉 Batch detection finished!")
    print(f"✅ Successfully processed: {success_count} images")
    print(f"❌ Failed/Skipped: {len(img_files) - success_count} images")
    print(f"📁 All results saved to: {folder_path}")

def main():
    """Main function: Single/batch detection mode (optimized version)"""
    print("="*60)
    print("      Lane Detection Program (Width Optimized Version)")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📂 Current script directory: {script_dir}")
    
    print("\nPlease select running mode:")
    print("1 - Single image detection (width optimized)")
    print("2 - Batch image detection (width optimized)")
    mode = input("Enter mode number (1/2): ").strip()
    
    if mode == "1":
        print("\n📸 Single image detection mode")
        print("\nPlease enter image relative path (relative to script directory):")
        print("Example: images/test_lane.png or ../data/test_lane.png")
        relative_path = input("Relative path: ").strip()
        
        if not relative_path:
            print("❌ Error: Path cannot be empty!")
            return
        
        CUSTOM_IMAGE_PATH = os.path.join(script_dir, relative_path)
        
        if not os.path.exists(CUSTOM_IMAGE_PATH):
            print(f"\n❌ Error: File does not exist!")
            print(f"Script directory: {script_dir}")
            print(f"Your input relative path: {relative_path}")
            print(f"Combined absolute path: {CUSTOM_IMAGE_PATH}")
            print("Please check path and filename")
            return
        
        img = cv2.imread(CUSTOM_IMAGE_PATH)
        if img is None:
            print("\n❌ Error: Cannot read image!")
            return
        
        print("\n✅ Image loaded successfully, detecting lanes...")
        result = lane_detection_pipeline(img)
        
        cv2.imshow("📷 Original Image", img)
        cv2.imshow("🚗 Lane Detection Result (Width Optimized)", result)
        
        save_path = os.path.splitext(CUSTOM_IMAGE_PATH)[0] + "_result.jpg"
        cv2.imwrite(save_path, result)
        print(f"\n✅ Detection completed! Result saved to: {save_path}")
        print("\nTip: Press any key to close image windows")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif mode == "2":
        print("\n📁 Batch image detection mode")
        print("\nPlease enter folder relative path (relative to script directory):")
        print("Example: lane_images or ../data/lane_images")
        relative_folder = input("Relative path: ").strip()
        
        if not relative_folder:
            print("❌ Error: Path cannot be empty!")
            return
        
        folder_path = os.path.join(script_dir, relative_folder)
        batch_detect_images(folder_path)
    
    else:
        print("❌ Error: Invalid mode number! Please enter 1 or 2")

if __name__ == "__main__":
    # Install dependencies before first run:
    # pip install opencv-python numpy scipy
    main()
