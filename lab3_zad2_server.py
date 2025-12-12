# zadanie2_server.py
import socket
import struct
import numpy as np
from PIL import Image


def send_all(sock, data):
    if isinstance(data, np.ndarray):
        height, width = data.shape[0], data.shape[1]
        channels = data.shape[2] if len(data.shape) == 3 else 1

        # Pakuj metadane (3 int32)
        metadata = struct.pack('iii', height, width, channels)

        # Pakuj dane pikseli
        pixel_data = data.tobytes()

        # Całość
        full_data = metadata + pixel_data

        # Wysyłaj długość (big-endian uint32) i dane
        length_bytes = struct.pack('>I', len(full_data))
        sock.sendall(length_bytes)
        sock.sendall(full_data)

        print(f"Wysłano {len(full_data)} bajtów ({height}x{width}x{channels})")


def receive_all(sock):
    # Odbierz długość (4 bajty, big-endian)
    length_bytes = sock.recv(4)
    if len(length_bytes) != 4:
        raise Exception("Nie udało się odczytać długości")

    length = struct.unpack('>I', length_bytes)[0]
    print(f"Oczekuję {length} bajtów...")

    # Odbierz dane
    data = b''
    while len(data) < length:
        packet = sock.recv(min(4096, length - len(data)))
        if not packet:
            break
        data += packet

    print(f"Odebrano {len(data)} bajtów")

    # Deserializuj metadane (3 int32 = 12 bajtów)
    height, width, channels = struct.unpack('iii', data[:12])
    pixel_data = data[12:]

    print(f"Wymiary: {height}x{width}x{channels}")

    # Odtwórz numpy array
    if channels == 1:
        result = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width)
    else:
        result = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)

    return result


def split_image(image, n_clients):
    img_array = np.array(image)
    height = img_array.shape[0]
    fragment_height = height // n_clients

    fragments = []
    for i in range(n_clients):
        start = i * fragment_height
        if i == n_clients - 1:  # ostatni fragment zawiera resztę
            end = height
        else:
            end = (i + 1) * fragment_height

        fragment = img_array[start:end, :, :]
        fragments.append(fragment)
        print(f"Fragment {i + 1}: {fragment.shape}")

    return fragments


def merge_image(processed_fragments):
    # Fragmenty są grayscale (2D), musimy dodać wymiar kanału
    fragments_with_channel = []
    for frag in processed_fragments:
        if len(frag.shape) == 2:
            # Dodaj wymiar kanału (height, width) -> (height, width, 1)
            frag = np.expand_dims(frag, axis=2)
        fragments_with_channel.append(frag)

    # Łączenie fragmentów pionowo
    merged = np.vstack(fragments_with_channel)

    # Jeśli jest tylko 1 kanał, usuń go dla PIL
    if merged.shape[2] == 1:
        merged = merged.squeeze(axis=2)

    return Image.fromarray(merged)


def server_main(image_path, n_clients, host='0.0.0.0', port=2040):
    print("=" * 60)
    print("SERWER PRZETWARZANIA OBRAZÓW")
    print("=" * 60)

    print("\nWczytywanie obrazu...")
    image = Image.open(image_path)
    print(f"Wczytano obraz: {image.size[0]}x{image.size[1]} pikseli")

    print(f"\nDzielenie obrazu na {n_clients} fragmentów...")
    fragments = split_image(image, n_clients)

    # Tworzenie gniazda serwera
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(n_clients)

    print(f"\nSerwer nasłuchuje na {host}:{port}...")
    print(f"Oczekuję połączeń od {n_clients} klientów...\n")

    processed_fragments = []

    for i in range(n_clients):
        print(f"{'=' * 60}")
        print(f"Oczekiwanie na klienta {i + 1}/{n_clients}...")

        client_socket, client_address = server_socket.accept()
        print(f"Połączono z klientem {i + 1}: {client_address}")

        # Wysyłanie fragmentu do klienta
        print(f"Wysyłanie fragmentu {i + 1} do klienta...")
        send_all(client_socket, fragments[i])

        # Odbieranie przetworzonego fragmentu
        print(f"Odbieranie przetworzonego fragmentu {i + 1}...")
        processed_fragment = receive_all(client_socket)
        processed_fragments.append(processed_fragment)

        client_socket.close()
        print(f"Zakończono przetwarzanie fragmentu {i + 1}\n")

    server_socket.close()

    print("=" * 60)
    print("Scalanie fragmentów...")
    result_image = merge_image(processed_fragments)

    print("Zapisywanie wyniku...")
    result_image.save("processed_image.png")
    print("Obraz przetworzony zapisany jako processed_image.png")
    print("=" * 60)
    print("ZAKOŃCZONO POMYŚLNIE!")
    print("=" * 60)


if __name__ == "__main__":
    IMAGE_PATH = "input_image.jpg"  # Ścieżka do obrazu wejściowego
    N_CLIENTS = 2  # Liczba klientów
    HOST = '0.0.0.0'  # Nasłuchuj na wszystkich interfejsach
    PORT = 2040  # Port

    try:
        server_main(IMAGE_PATH, N_CLIENTS, HOST, PORT)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{IMAGE_PATH}'")
        print(f"   Upewnij się że plik znajduje się w tym samym folderze co serwer!")
    except KeyboardInterrupt:
        print("\n\nPrzerwano działanie serwera (Ctrl+C)")
    except Exception as e:
        print(f"\nBŁĄD: {e}")
        import traceback

        traceback.print_exc()