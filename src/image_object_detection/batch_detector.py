# batch_detector.py

import os
import cv2
from pathlib import Path
from detection_engine import DetectionEngine, ModelLoadError


class BatchDetector:
    def __init__(self, detection_engine, input_dir, output_dir):
        
        self.engine = detection_engine
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

    def run(self):
        image_files = [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_extensions
        ]

        if not image_files:
            print(f"⚠️ No valid image files found in {self.input_dir}")
            return

        print(f"🔍 Found {len(image_files)} images. Starting batch detection...")
        success_count = 0

        for img_path in sorted(image_files):
            try:
                # 读取图像
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"❌ Failed to read image (corrupted or unsupported): {img_path.name}")
                    continue

                annotated_frame, _ = self.engine.detect(frame)

                output_path = self.output_dir / f"{img_path.stem}_detected{img_path.suffix}"

                if cv2.imwrite(str(output_path), annotated_frame):
                    print(f"✅ Saved: {output_path.name}")
                    success_count += 1
                else:
                    print(f"❌ Failed to save: {output_path}")

            except Exception as e:
                print(f"💥 Error processing {img_path.name}: {e}")

        print(f"\n🎉 Batch detection completed. {success_count}/{len(image_files)} images processed successfully.")
