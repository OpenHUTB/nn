from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from os.path import join

class Lane:
    """Detects Lanes in a given Image
    """
    HLS_L_THRESHOLD_MIN = 150
    HLS_L_THRESHOLD_MAX = 255
    XM_PER_PIX = 3.7 / 781
    ROI_BOTTOM = 350
    ROI_TOP = 260
    ROI_LEFT = 200
    ROI_RIGHT = 440
    # === 新增 ===
    MARGIN_RATIO = 1.0 / 12.0
    # ===========
    def __init__(self, height:int, width:int, save:bool=False,
        save_folder:str=join('img', 'examples')) -> None:
        """Constructor

        Args:
            height (int): Height of the Image.
            width (int): Width of the Image.
            save (bool, optional): Whether to save the Images that will be
                generated during the process. Defaults to False.
            save_folder (str, optional): Folder to save the Images. This only
                applies if save is true. Defaults to join('img', 'examples').
        """
        self.save = save
        self.save_folder = save_folder
        self.img_height = height
        self.img_width = width
        self.margin = int(self.MARGIN_RATIO * self.img_width)
        self.DEGREE = 2

        # Variables to store for lines
        self.x_left = None
        self.x_right = None
        self.y_left = None
        self.y_right = None

        # Variables for fill
        self.x_left_fill = None
        self.x_right_fill = None
        self.y_left_fill = None
        self.y_right_fill = None

    def get_lines(self, img:np.ndarray) -> np.ndarray:
        """Get white Lines

        Filters out lighter parts of the image.

        Args:
            img (np.ndarray): Image on which the changes are to be applied.

        Returns:
            np.ndarray: Image with changes applied.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Create threshold matrix to differentiate black and white based on the
        # lightness (of HSL).
        _, binary = cv2.threshold(hls[:, :, 1], self.HLS_L_THRESHOLD_MIN, self.HLS_L_THRESHOLD_MAX, cv2.THRESH_BINARY)

        # [Perception] Apply Morphological Closing to fill gaps in lane lines
        # and remove small noise dots.
        kernel = np.ones((5, 5), np.uint8)
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Blur the processed image
        binary_blured = cv2.GaussianBlur(binary_closed, (3, 3), 0)

        if self.save:
            cv2.imwrite(join(self.save_folder, 'lines.jpg'), binary_blured)

        return binary_blured

    def extract_roi(self, img: np.ndarray,
                    bottom: int = ROI_BOTTOM,
                    top: int = ROI_TOP,
                    left: int = ROI_LEFT,
                    right: int = ROI_RIGHT) -> tuple[np.ndarray, np.ndarray]:
        """Extract the Region of Interest

        Extracts the region of interest and transforms it to fit the original
        image size.

        Args:
            img (np.ndarray): Image on which the changes are to be applied.
                Should generally be returned by get_lines.
            bottom (int, optional): Bottom of the trapezoid. Defaults to 350.
            top (int, optional): Top of the trapezoid. Defaults to 260.
            left (int, optional): Top-Left of the trapezoid. Defaults to 200.
            right (int, optional): Top-Right of the trapezoid. Defaults to 440.

        Returns:
            np.ndarray: Image with changes applied.
            tuple[np.ndarray, np.ndarray]:
                [0]: Image showing the warped perspective.
                [1]: Matrix representing the inverse operation to undo the
                    transformation.
        """
        # Corners of the trapezoidal relevant area.
        roi_points = np.float32([
            [left, top],
            [0, bottom], # Bottom-Left            
            [self.img_width, bottom], # Bottom-Right
            [right, top]
        ])

        # Padding to apply on the transformed image.
        padding = int(0.25 * self.img_width)
        # Corners of the trapezoidal relevant area after transformation.
        desired_roi_points = np.float32([
            [padding, 0], # Top-left corner
            [padding, self.img_height], # Bottom-left corner         
            [self.img_width-padding, self.img_height], # Bottom-right corner
            [self.img_width-padding, 0] # Top-right corner
        ])

        perspective_transform = cv2.getPerspectiveTransform(roi_points,
            desired_roi_points)
        inverse_perspective_transform = cv2 \
            .getPerspectiveTransform(desired_roi_points, roi_points)
        perspective = cv2.warpPerspective(img, perspective_transform,
            (self.img_width, self.img_height), flags=cv2.INTER_LINEAR)
        _, warped_perspective_binary = cv2.threshold(perspective, 127, 255,
            cv2.THRESH_BINARY)

        if self.save:
            COLOR_TRAPEZE = (147,20,255)
            warped_perspective = cv2.polylines(warped_perspective_binary.copy(),
                np.int32([desired_roi_points]), True, COLOR_TRAPEZE, 3)
            cv2.imwrite(join(self.save_folder, 'warped_perspective.jpg'),
                warped_perspective)
        
        return warped_perspective_binary, inverse_perspective_transform

    def get_hist(self,
        img:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates Historgramm of light Parts inside the image.

        Args:
            img (np.ndarray): Image on which the changes are to be applied.
                Should generally be returned by extract_roi.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                [0]: Values of the histogram.
                [1]: Width-index of the location of the left lane.
                [2]: Width-index of the location of the right lane.
        """
        hist = np.sum(img[int(img.shape[0]/2):,:], axis=0)
        midpoint = np.int(hist.shape[0]/2)
        left_lane = np.argmax(hist[:midpoint])
        right_lane = np.argmax(hist[midpoint:]) + midpoint

        if self.save:
            _, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
            plt.subplots_adjust(hspace=0.25)
            ax1.imshow(img, cmap='gray', aspect='auto')
            ax1.set_title('Transformation of the Binary Representation')
            ax2.plot(hist)
            ax2.set_title('Histogram of the Deflections')
            plt.savefig(join(self.save_folder, 'warped_hist.jpg'))

        return hist, left_lane, right_lane

    def get_line_fits(self, img:np.ndarray, max_left_idx:np.ndarray,
        max_right_idx:np.ndarray,
        number_windows:int=10) -> tuple[np.ndarray, np.ndarray]:
        """Apply Sliding Windows and Retrieve Lines

        Args:
            img (np.ndarray): Image on which the changes are to be applied.
                Should generally be the original image.
            max_left_idx (np.ndarray): Index of the detected left line as
                returned by get_hist.
            max_right_idx (np.ndarray): Index of the detected right line as
                returned by get_hist.
            number_windows (int, optional): Number of windows to create.
                Defaults to 10.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                [0]: Second order polynomial curve to represent the left line.
                [1]: Second order polynomial curve to represent the right line.
        """
        # Number of pixels to identify line.
        MINIMAL_AREA = int((1/24) * self.img_width)
        WINDOW_HEIGHT = np.int(img.shape[0]/number_windows)       
    
        # Retrive x and y coordinates of white pixels (non-zero).
        nonzero_y, nonzero_x = img.nonzero()
            
        # Store the pixel indices for the left and right lane lines
        left_lane_idxs = []
        right_lane_idxs = []
    
        WINDOW_COLOR = (255,255,255)
        WINDOW_THICKNESS = 2

        for window in range(number_windows):
            y_low = img.shape[0] - (window + 1) * WINDOW_HEIGHT
            y_high = img.shape[0] - window * WINDOW_HEIGHT
            x_low_left = max_left_idx - self.margin
            x_high_left = max_left_idx + self.margin
            x_low_right = max_right_idx - self.margin
            x_high_right = max_right_idx + self.margin

            # Draw windows.
            cv2.rectangle(img, (x_low_left, y_low), (x_high_left, y_high),
                WINDOW_COLOR, WINDOW_THICKNESS)
            cv2.rectangle(img, (x_low_right, y_low),(x_high_right, y_high),
                WINDOW_COLOR, WINDOW_THICKNESS)
        
            # Find the white pixels in x and y inside the window.
            non_zero_left_idxs = ((nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= x_low_left) & (nonzero_x < x_high_left)) \
                .nonzero()[0]
            non_zero_right_idxs = ((nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= x_low_right) & (nonzero_x < x_high_right)) \
                .nonzero()[0]
            left_lane_idxs.append(non_zero_left_idxs)
            right_lane_idxs.append(non_zero_right_idxs)
                
            # Recenter next window on mean position.
            if len(non_zero_left_idxs) > MINIMAL_AREA:
                max_left_idx = np.int(np.mean(nonzero_x[non_zero_left_idxs]))
            if len(non_zero_right_idxs) > MINIMAL_AREA:        
                max_right_idx = np.int(np.mean(nonzero_x[non_zero_right_idxs]))
                        
        left_lane_idxs = np.concatenate(left_lane_idxs)
        right_lane_idxs = np.concatenate(right_lane_idxs)

        # If valid lane has been found, update values for next image. This
        # ensures that if no lane has been found, the detection within the next
        # image are still possible. However, this requires to detect a lane
        # within the first image after the lane object has been created.     
        is_invalid = (sum(nonzero_x[left_lane_idxs]) == 0) or \
            (sum(nonzero_y[left_lane_idxs]) == 0)
        is_false = (nonzero_x[left_lane_idxs].size == 0) or \
            (nonzero_x[left_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.x_left = nonzero_x[left_lane_idxs].copy()
        
        is_false = (nonzero_y[left_lane_idxs].size == 0) or \
            (nonzero_y[left_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.y_left = nonzero_y[left_lane_idxs].copy()

        is_invalid = (sum(nonzero_x[right_lane_idxs]) == 0) or \
            (sum(nonzero_y[right_lane_idxs]) == 0)
        is_false = (nonzero_x[right_lane_idxs].size == 0) or \
            (nonzero_x[right_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.x_right = nonzero_x[right_lane_idxs].copy()
        
        is_false = (nonzero_y[right_lane_idxs].size == 0) or \
            (nonzero_y[right_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.y_right = nonzero_y[right_lane_idxs].copy()
    
        # Fit a second order polynomial curve to represent the lines.
        left_line = np.polyfit(self.y_left, self.x_left, self.DEGREE)
        right_line = np.polyfit(self.y_right, self.x_right, self.DEGREE)

        # Retrieve x and y coordinates by applying the ploynomical curves.
        y = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fit_x = left_line[0]*y**2 + left_line[1]*y + \
            left_line[2]
        right_fit_x = right_line[0]*y**2 + right_line[1]*y + \
            right_line[2]

        if self.save:
            out_img = np.dstack((img, img, img)) * 255

            left_lane_idxs = ((nonzero_x > (left_line[0]*(nonzero_y**2) +
                left_line[1]*nonzero_y + left_line[2] - self.margin)) &
                (nonzero_x < (left_line[0]*(nonzero_y**2) +
                left_line[1]*nonzero_y + left_line[2] + self.margin))) 
            right_lane_idxs = ((nonzero_x > (right_line[0]*(nonzero_y**2) +
                right_line[1]*nonzero_y + right_line[2] - self.margin)) &
                (nonzero_x < (right_line[0]*(nonzero_y**2) +
                right_line[1]*nonzero_y + right_line[2] + self.margin)))
                        
            # Apply color to the left and right line pixels.
            out_img[nonzero_y[left_lane_idxs], nonzero_x[left_lane_idxs]] = \
                [255, 0, 0]
            out_img[nonzero_y[right_lane_idxs], nonzero_x[right_lane_idxs]] = \
                [0, 0, 255]

            figure, (ax1, ax2) = plt.subplots(2)
            figure.tight_layout(pad=3.0)
            ax1.imshow(img, cmap='gray')
            ax2.imshow(out_img)
            ax2.plot(left_fit_x, y, color='yellow')
            ax2.plot(right_fit_x, y, color='yellow')
            ax1.set_title('Sliding Windows of the Transformed Image')
            ax2.set_title('Detected Track Lines')
            plt.savefig(join(self.save_folder, 'sliding_windows.jpg'))
        
        return left_line, right_line

    def get_search_window(self, img:np.ndarray, left_line:np.ndarray,
        right_line:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the Search Window for Lines

        Args:
            img (np.ndarray): Image on which the changes are to be applied.
                Should generally be the original image.
            left_line (np.ndarray): Second order polynomial curve to represent
                the left line.
            right_line (np.ndarray): Second order polynomial curve to represent
                the right line.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                [0]: Values returned by applying the polynomial curve to using
                    the y values for the left line.
                [1]: Values returned by applying the polynomial curve to using
                    the y values for the right line.
                [2]: Generated y values to get a continuous look.
        """
        # Retrive x and y coordinates of white pixels (non-zero).       
        nonzero_y, nonzero_x = img.nonzero()
            
        # Get indices for he left and right line pixels.
        left_lane_idxs = ((nonzero_x > (left_line[0]*(nonzero_y**2) +
            left_line[1]*nonzero_y + left_line[2] - self.margin)) &
            (nonzero_x < (left_line[0]*(nonzero_y**2) + left_line[1]*nonzero_y +
            left_line[2] + self.margin))) 
        right_lane_idxs = ((nonzero_x > (right_line[0]*(nonzero_y**2) +
            right_line[1]*nonzero_y + right_line[2] - self.margin)) &
            (nonzero_x < (right_line[0]*(nonzero_y**2) +
            right_line[1]*nonzero_y + right_line[2] + self.margin)))           

        # If valid lane has been found, update values for next image. This
        # ensures that if no lane has been found, the detection within the next
        # image are still possible. However, this requires to detect a lane
        # within the first image after the lane object has been created.  
        is_invalid = (sum(nonzero_x[left_lane_idxs]) == 0) or \
            (sum(nonzero_y[left_lane_idxs]) == 0)
        is_false = (nonzero_x[left_lane_idxs].size == 0) or \
            (nonzero_x[left_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.x_left_fill = nonzero_x[left_lane_idxs].copy()
        
        is_false = (nonzero_y[left_lane_idxs].size == 0) or \
            (nonzero_y[left_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.y_left_fill = nonzero_y[left_lane_idxs].copy()

        is_invalid = (sum(nonzero_x[right_lane_idxs]) == 0) or \
            (sum(nonzero_y[right_lane_idxs]) == 0)
        is_false = (nonzero_x[right_lane_idxs].size == 0) or \
            (nonzero_x[right_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.x_right_fill = nonzero_x[right_lane_idxs].copy()
        
        is_false = (nonzero_y[right_lane_idxs].size == 0) or \
            (nonzero_y[right_lane_idxs] is None)
        if not (is_false or is_invalid):
            self.y_right_fill = nonzero_y[right_lane_idxs].copy()

        # Fit a second order polynomial curve to represent the outer lines.
        left_fit = np.polyfit(self.y_left_fill, self.x_left_fill, self.DEGREE)
        right_fit = np.polyfit(self.y_right_fill, self.x_right_fill,
            self.DEGREE)
            
        # Retrieve x and y coordinates by applying the ploynomical curves.
        y = np.linspace(0, img.shape[0]-1, img.shape[0]) 
        left_fit_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        right_fit_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
            
        if self.save:
            out_img = np.dstack((img, img, img))*255
            window_img = np.zeros_like(out_img)
                    
            # Apply color to the left and right line pixels.
            out_img[nonzero_y[left_lane_idxs], nonzero_x[left_lane_idxs]] = \
                [255, 0, 0]
            out_img[nonzero_y[right_lane_idxs], nonzero_x[right_lane_idxs]] = \
                [0, 0, 255]

            # Create a polygon to represent the search window area and convert
            # the x and y points into a usable format for cv2.fillPoly().
            left_line_window1 = np.array([np.transpose(
                np.vstack([left_fit_x-self.margin, y]))])
            left_line_window2 = np.array([np.flipud(np.transpose(
                np.vstack([left_fit_x+self.margin, y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(
                np.vstack([right_fit_x-self.margin, y]))])
            right_line_window2 = np.array([np.flipud(np.transpose(
                np.vstack([right_fit_x+self.margin, y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
                    
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            
            _, ax= plt.subplots(1)
            ax.imshow(result)
            ax.plot(left_fit_x, y, color='yellow')
            ax.plot(right_fit_x, y, color='yellow')
            ax.set_title('Search Window on the Transformed Image')
            plt.savefig(join(self.save_folder, 'search_window.jpg'))

        return left_fit_x, right_fit_x, y

    def show_lane(self, orig_img:np.ndarray, img:np.ndarray,
        left_fit_x:np.ndarray, right_fit_x:np.ndarray, y:np.ndarray,
        inverse_perspective_transform:np.ndarray) -> tuple[np.ndarray, int]:
        """Draw Lane on original Image

        Args:
            orig_img (np.ndarray): Original image before any changes.
            img (np.ndarray): Image after warping.
            left_fit_x (np.ndarray): Values returned by applying the polynomial
                curve to using the y values for the left line.
            right_fit_x (np.ndarray): Values returned by applying the polynomial
                curve to using the y values for the right line.
            y (np.ndarray): Generated y values to get a continuous look.
            inverse_perspective_transform (np.ndarray): Matrix representing the
                inverse operation to undo the transformation.

        Returns:
            tuple[np.ndarray, int]:
                [0]: Original image with lane drawn on.
                [1]: Detected surface area multiplied by 255.
        """

        # Generate image to draw on.
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       
            
        # Converting the x and y points into a usable format for cv2.fillPoly().
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, y]))])
        pts_right = np.array([np.flipud(np.transpose(
            np.vstack([right_fit_x, y])))])
        pts = np.hstack((pts_left, pts_right))
            
        # Draw lane on warped image.
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
        # Undo warp transformation.
        newwarp = cv2.warpPerspective(color_warp, inverse_perspective_transform,
            (orig_img.shape[1], orig_img.shape[0]))
        
        # Draw lane on original image.
        result = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)
            
        if self.save:
            _, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax.set_title('Original Image with Recognised Track')
            plt.savefig(join(self.save_folder, 'lane_overlay.jpg'))

        return result, np.sum(color_warp)

    def get_car_position(self, img:np.ndarray, left_line:np.ndarray,
        right_line:np.ndarray) -> tuple[np.ndarray, float]:
        """Retrieve the Error

        Args:
            img (np.ndarray): Original image with lane drawn on.
            left_line (np.ndarray): Second order polynomial curve to represent
                the left line.
            right_line (np.ndarray): Second order polynomial curve to represent
                the right line.

        Returns:
            tuple[np.ndarray, float]:
                [0]: Original image with lane and offset drawn on.
                [1]: Calculated offset to the center of the detected lane.
        """

        # Retrieve the position of the car assuming it is the center of the
        # image. 
        car_location = self.img_width / 2
    
        # Retrieve x coordinates of bottom from the lane.
        bottom_left = left_line[0]*self.img_height**2 + \
            left_line[1]*self.img_height + left_line[2]
        bottom_right = right_line[0]*self.img_height**2 + \
            right_line[1]*self.img_height + right_line[2]
    
        center_lane = (bottom_right - bottom_left)/2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(center_lane)) * \
                        self.XM_PER_PIX * 100
        
        cv2.putText(img, (f'Difference to Center: '
            f'{str(round(center_offset, 5))} cm'), (int((5/600)*self.img_width),
            int((20/338)*self.img_height)), cv2.FONT_HERSHEY_SIMPLEX,
            (float((0.5/600)*self.img_width)), (255,255,255),2,cv2.LINE_AA)

        return img, center_offset

    def pipe(self, img:np.ndarray) -> tuple[np.ndarray, float, int]:
        """Pipes Image through every Step in right order.

        Args:
            img (np.ndarray): Original image to detect lane on.

        Returns:
            tuple[np.ndarray, float, int]:
                [0]: Original image with lane and offset drawn on.
                [1]: Calculated offset to the center of the detected lane.
                [2]: Detected surface area multiplied by 255.
        """
        original_image = img.copy()
        img = self.get_lines(img)
        img, inverse_perspective_transform = self.extract_roi(img)
        _, max_left_idx, max_right_idx = self.get_hist(img)
        left_line, right_line = self.get_line_fits(img.copy(),
            max_right_idx, max_left_idx)
        left_fit_x, right_fit_x, y = self.get_search_window(img.copy(),
            left_line, right_line)
        img, detection_surface_area = self.show_lane(original_image,
            img.copy(), left_fit_x, right_fit_x, y,
            inverse_perspective_transform)
        img, error = self.get_car_position(img, left_line, right_line)
        return img, error, detection_surface_area