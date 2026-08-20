import cv2


def draw_box(img, box, label):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        img,
        str(label),
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )