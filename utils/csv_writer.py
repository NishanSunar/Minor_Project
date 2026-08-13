import csv


class CSVWriter:

    def __init__(self, filename="output.csv"):
        self.filename = filename

        self.file = open(
            self.filename,
            "w",
            newline="",
            encoding="utf-8-sig"
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "Frame",
            "Vehicle ID",
            "Plate",
            "Timestamp",
            "Location",
            "Confidence"
        ])

    def write(
        self,
        frame_number,
        vehicle_id,
        plate,
        timestamp,
        location,
        confidence
    ):
        self.writer.writerow([
            frame_number,
            vehicle_id,
            plate,
            timestamp,
            location,
            confidence
        ])

        # Make sure data is immediately written
        self.file.flush()

    def close(self):
        self.file.close()