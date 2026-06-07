import os
import cv2
import numpy as np

# Define the input directory and output directory
input_dir = 'data/data-frames/data-frames-3fps'
output_dir = 'data/collages/collages-3fps-2-2'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all subfolders in the main directory
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Loop through the next level of subfolders
        for inner_subfolder in os.listdir(subfolder_path):
            inner_subfolder_path = os.path.join(subfolder_path, inner_subfolder)
            if os.path.isdir(inner_subfolder_path):
                # Get all frame files within this inner subfolder
                frame_files = sorted(os.listdir(inner_subfolder_path))
                
                # Calculate the number of collages needed
                num_collages = len(frame_files) // 4 + (1 if len(frame_files) % 4 > 0 else 0)
                
                # Determine the padding width based on the number of collages
                padding_width = len(str(num_collages))

                # Loop through each set of frames (up to 4 frames)
                for collage_index in range(num_collages):
                    # Get the current set of frames (up to 4 frames)
                    frames = []
                    for i in range(4):
                        if collage_index * 4 + i < len(frame_files):
                            frame_path = os.path.join(inner_subfolder_path, frame_files[collage_index * 4 + i])
                            frame = cv2.imread(frame_path)
                            frames.append(frame)

                    # Create collage only if there are frames
                    if frames:
                        # Determine the size of the collage
                        frame_height, frame_width = frames[0].shape[:2]
                        padding = 15
                        collage_height = 2 * frame_height + 3 * padding
                        collage_width = 2 * frame_width + 3 * padding

                        # Create a blank image for the collage with padding
                        collage = np.full((collage_height, collage_width, 3), 255, dtype=np.uint8)  # White background

                        # Place frames in the collage with padding
                        positions = [
                            (padding, padding),  # Top-left
                            (padding, 2 * padding + frame_width),  # Top-right
                            (2 * padding + frame_height, padding),  # Bottom-left
                            (2 * padding + frame_height, 2 * padding + frame_width)  # Bottom-right
                        ]

                        for i, (frame, pos) in enumerate(zip(frames, positions)):
                            y, x = pos
                            collage[y:y+frame_height, x:x+frame_width] = frame
                            # Add black highlight for text
                            cv2.rectangle(collage, (x, y), (x + 80, y + 50), (0, 0, 0), -1)
                            # Add frame number
                            cv2.putText(
                                collage,
                                f'{collage_index * 4 + i + 1}',  # Frame number
                                (x + 10, y + 40),  # Position
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),  # White color for text
                                2,
                                cv2.LINE_AA  # Thickness
                            )

                        # Save the collage preserving the subfolder structure
                        output_subfolder = os.path.join(output_dir, subfolder, inner_subfolder)
                        os.makedirs(output_subfolder, exist_ok=True)
                        output_path = os.path.join(output_subfolder, f'collage_{collage_index + 1:0{padding_width}d}.jpg')
                        cv2.imwrite(output_path, collage)
                        print(f"Saved collages to '{output_path}'")
