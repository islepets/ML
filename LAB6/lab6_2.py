import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def preprocess_image(image_path, img_size=(150, 150)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    model = load_model('face_mask_model.h5')

    numper_picture = input("Введите номер картинки: ")
    image_path = f"C:/Users/ilya_slepets/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/images/maksssksksss{numper_picture}.png"

    img_array = preprocess_image(image_path, img_size=(150, 150))

    prediction = model.predict(img_array, verbose=0)

    probability_no_mask = prediction[0][0]
    probability_with_mask = 1 - probability_no_mask

    if probability_no_mask > 0.5:
        predicted_class = "Без маски"
        confidence = probability_no_mask
    else:
        predicted_class = "С маской"
        confidence = probability_with_mask

    print(f"Результат классификации: {predicted_class}")
    print(f"Уверенность: {confidence:.2%}")
    print(f"Вероятность 'С маской': {probability_with_mask:.4f} ({probability_with_mask:.2%})")
    print(f"Вероятность 'Без маски': {probability_no_mask:.4f} ({probability_no_mask:.2%})")

if __name__ == '__main__':
    main()