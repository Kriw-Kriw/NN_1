import numpy as np
from PIL import Image
import os

GRID_SIZE = 20  # размер изображения
SPEED = 0.45  # скорость обучения
EPOCHS = 3000  # количество эпох


# инициализация весов
def initialize_weights(grid_size: int):
    return np.random.uniform(-0.3, 0.3, (grid_size, grid_size))


# перевод изображения в матрицу
def load_image(path: str):
    img = Image.open(path).convert('L')
    img = img.resize((GRID_SIZE, GRID_SIZE))
    img_matrix = np.asarray(img) // 255
    return img_matrix


# уменьшаем изображение
def scale_image(img_matrix):
    non_zero_rows = np.any(img_matrix, axis=1)
    non_zero_cols = np.any(img_matrix, axis=0)
    cropped_img = img_matrix[np.ix_(non_zero_rows, non_zero_cols)]
    scaled_img = Image.fromarray(cropped_img * 255).resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(scaled_img) // 255


# функция активации
def activation_function(img_matrix, weights):
    total_sum = np.sum(img_matrix * weights)
    return 1 if total_sum > 0 else 0


def train_perceptron(img_matrix, weights, target):
    output = activation_function(img_matrix, weights)
    error = target - output
    weights += SPEED * error * img_matrix
    return weights, error


def save_weights(weights, filename='weights.npy'):
    np.save(filename, weights)


def load_weights(filename='weights.npy'):
    return np.load(filename)


def load_images_from_folder(path: str, label: int):
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img_matrix = load_image(os.path.join(path, filename))
            img_matrix = scale_image(img_matrix)
            images.append(img_matrix)
            labels.append(label)
    return images, labels


def train_on_images(path1: str, path2: str):
    images1, labels1 = load_images_from_folder(path1, 0)
    images2, labels2 = load_images_from_folder(path2, 1)

    images = images1 + images2
    labels = labels1 + labels2
    try:
        weights = load_weights('weights.npy')
        print('Есть сохранённые веса')
    except FileNotFoundError:
        print('Нет сохранённых весов. Инициализация новых')
        weights = initialize_weights(GRID_SIZE)
    for epoch in range(EPOCHS):
        total_error = 0
        for img_matrix, label in zip(images, labels):
            weights, error = train_perceptron(img_matrix, weights, label)
            total_error += abs(error)  # средняя ошибка
        print(f'Эпоха {epoch + 1}/{EPOCHS}, Ошибки: {total_error}')

    save_weights(weights)
    print('Веса сохранены')

    return weights


def predict_image(path: str, trained_weights):
    test_image = load_image(path)
    test_image = scale_image(test_image)
    result = activation_function(test_image, trained_weights)
    print(f'Предсказание для тестового изображения {path}: {result} (1 - Y; 0 - X)')


def result(path: str, trained_weights):
    for filename in os.listdir(path):
        image = os.path.join(path, filename)
        predict_image(image, trained_weights)


if __name__ == "__main__":
    path1 = r'images/train/X'
    path2 = r'images/train/Y'

    trained_weights = train_on_images(path1, path2)
    result(r'images/test', trained_weights)
