import cv2


class ImagePreprocessor:
    def preprocess(self, image):
        if image is None or image.size == 0:
            return image

        image = cv2.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        return image