import cv2
import numpy as np

__all__ = ["ContourParser"]


class ContourParser:
    def __init__(self, image):
        if hasattr(image, "size") and image.size == 0:
            raise ValueError(
                f"The image of type {image.__class__.__name__} \
                is empty. Expected 'np.ndarray' object."
            )
        if image is None:
            raise ValueError("The image passed in 'NoneType'")
        self.image = image

    def apply_threshold(self):
        grey = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

        ret, threshhold = cv2.threshold(
            grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)

        return threshhold

    def find_contours(self):
        thresh = self.apply_threshold()

        contours = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        preprocessed_digits = []

        for c in contours[0]:
            """
                Creating a rectangle around the digit
                in the original image (for displaying
                the digits fetched via contours)
            """
            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(
                self.image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2
            )

            """
                Cropping out the digit from the image
                corresponding to the current contours
                in the for loop
            """

            digit = thresh[y: y + h, x: x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18, 18))

            """
                Padding the digit with 5 pixels of black
                color (zeros) in each side to finally
                produce the image of (28, 28)
            """
            padded_digit = np.pad(
                resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0
            )

            # Adding the preprocessed digit to the list of preprocessed digits
            preprocessed_digits.append(padded_digit)

            return preprocessed_digits
