import cv2


class ImagePreprocessor:

    def preprocess(self, plate_image):
        """
        Preprocess license plate image for OCR.
        """

        # 1. Resize
        height, width = plate_image.shape[:2]

        new_width = 200
        scale = new_width / width
        new_height = int(height * scale)

        resized = cv2.resize(plate_image,(new_width, new_height),interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        # 3. Reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # 4. Binary threshold
        _, thresh = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
      
        return thresh
    