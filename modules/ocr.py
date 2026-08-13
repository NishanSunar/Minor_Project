import easyocr


class OCRReader:

    def __init__(self):
        """
        Load EasyOCR model once.
        """
        self.reader =easyocr.Reader(['en','ne'],gpu=False)

    def read_text(self, plate_image):
        """
        Extract text from license plate image.

        Returns:
            EasyOCR results
        """

        results = self.reader.readtext(
            plate_image
        )

        return results