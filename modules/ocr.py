import cv2
from paddleocr import PaddleOCR


class OCRReader:

    def __init__(self):

        # English OCR
        self.english = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

        # Devanagari OCR
        self.devanagari = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    # --------------------------------------------------------
    # RUN ONE OCR MODEL
    # --------------------------------------------------------

    def _run(self, model, image):

        if image is None or image.size == 0:
            return []

        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        try:
            results = model.predict(image)

        except Exception as e:
            print(
                f"OCR model error: "
                f"{type(e).__name__}: {e}"
            )
            return []

        output = []

        for result in results:

            try:

                if hasattr(result, "json"):
                    result = result.json

                data = (
                    result["res"]
                    if (
                        isinstance(result, dict)
                        and "res" in result
                    )
                    else result
                )

                if not isinstance(data, dict):
                    continue

                texts = data.get(
                    "rec_texts",
                    []
                )

                scores = data.get(
                    "rec_scores",
                    []
                )

                boxes = data.get(
                    "rec_boxes",
                    []
                )

                for i, (text, score) in enumerate(
                    zip(texts, scores)
                ):

                    try:
                        score = float(score)
                    except:
                        continue

                    text = str(text).strip()

                    if not text:
                        continue

                    box = None

                    if i < len(boxes):
                        try:
                            box = boxes[i].tolist()
                        except:
                            box = None

                    output.append({
                        "text": text,
                        "confidence": score,
                        "box": box
                    })

            except Exception:
                continue

        return output

    # --------------------------------------------------------
    # MAIN OCR
    # --------------------------------------------------------

    def read_text(self, image):

        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if w <= 0 or h <= 0:
            return []

        results = []

        aspect = w / float(h)

        # ====================================================
        # TWO-LINE / TALL NEPALI PLATE
        # ====================================================

        if aspect < 2.8:

            split = int(h * 0.48)

            top = image[:split, :]
            bottom = image[split:, :]

            # Devanagari on upper part
            dev_results = self._run(
                self.devanagari,
                top
            )

            for item in dev_results:
                item["position"] = "top"
                results.append(item)

            # English/numbers on lower part
            eng_results = self._run(
                self.english,
                bottom
            )

            for item in eng_results:
                item["position"] = "bottom"
                results.append(item)

            # Also run full image as fallback
            full_dev = self._run(
                self.devanagari,
                image
            )

            for item in full_dev:
                item["position"] = "full-dev"
                results.append(item)

            full_eng = self._run(
                self.english,
                image
            )

            for item in full_eng:
                item["position"] = "full-eng"
                results.append(item)

        # ====================================================
        # ONE-LINE PLATE
        # ====================================================

        else:

            eng_results = self._run(
                self.english,
                image
            )

            for item in eng_results:
                item["position"] = "full-eng"
                results.append(item)

            # Devanagari fallback
            dev_results = self._run(
                self.devanagari,
                image
            )

            for item in dev_results:
                item["position"] = "full-dev"
                results.append(item)

        return results