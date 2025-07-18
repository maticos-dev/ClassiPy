import numpy as np

__all__ = ["DigitLoader"]


class DigitLoader:
    def __init__(self, preprocessed_digits) -> None:
        assert (
            isinstance(preprocessed_digits, np.ndarray),
            "Please pass a numpy array of pixels to represent each image",
        )

        self.digits = preprocessed_digits

    def infer_images(self, model):
        for digit in self.digits:
            assert (
                hasattr(model, "predict"),
                "Please pass a model with inference capabilities. 'predict' method not found",
            )

            prediction = model.predict(digit.reshape(1, 28 * 28))

            # plt.imshow(digit.reshape(28, 28), cmap="gray")
            # plt.show()

            print("\nPrediction from the neural network: {}".format(prediction))

            hard_maxed_prediction = np.zeros(prediction.shape)
            print("\n\n---------------------------------------\n\n")
