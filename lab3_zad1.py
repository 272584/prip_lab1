from multiprocessing import Pool
from PIL import Image
import numpy as np
import time


def edge_filter(image_array):
    if len(image_array.shape) == 3:
        gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image_array.copy()

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    height, width = gray.shape
    result = np.zeros_like(gray)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = gray[i - 1:i + 2, j - 1:j + 2]
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            result[i, j] = np.sqrt(gx ** 2 + gy ** 2)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def process_fragment(args):
    fragment_array, fragment_id = args
    print(f"Przetwarzanie fragmentu {fragment_id}...")
    processed = edge_filter(fragment_array)
    return processed, fragment_id


def split_image(image, n_parts):
    img_array = np.array(image)
    height = img_array.shape[0]
    fragment_height = height // n_parts

    fragments = []
    for i in range(n_parts):
        start = i * fragment_height
        if i == n_parts - 1:
            end = height
        else:
            end = (i + 1) * fragment_height

        fragment = img_array[start:end, :, :]
        fragments.append((fragment, i))

    return fragments


def merge_fragments(processed_fragments):
    processed_fragments.sort(key=lambda x: x[1])

    merged = np.vstack([frag[0] for frag in processed_fragments])
    return Image.fromarray(merged)


def main():
    image_path = "input_image.jpg"
    n_processes = 4

    print("Wczytywanie obrazu...")
    image = Image.open(image_path)

    print(f"Dzielenie obrazu na {n_processes} fragmentów...")
    fragments = split_image(image, n_processes)

    print("Równoległe przetwarzanie...")
    start_time = time.time()

    with Pool(n_processes) as pool:
        results = pool.map(process_fragment, fragments)

    end_time = time.time()

    print("Scalanie fragmentów...")
    result_image = merge_fragments(results)

    print("Zapisywanie wyniku...")
    result_image.save("processed_parallel.png")

    print(f"Zakończono! Czas przetwarzania: {end_time - start_time:.2f} sekund")


if __name__ == "__main__":
    main()