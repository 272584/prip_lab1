# zadanie2_client.py
import socket
import pickle
import numpy as np


def send_all(sock, data):
    """Wysyła pełne dane przez socket."""
    data = pickle.dumps(data)
    sock.sendall(len(data).to_bytes(4, byteorder='big'))
    sock.sendall(data)


def receive_all(sock):
    """Odbiera pełne dane przez socket."""
    length = int.from_bytes(sock.recv(4), byteorder='big')
    data = b''
    while len(data) < length:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
    return pickle.loads(data)


def edge_filter(image_array):
    """
    Implementacja filtru Sobela.
    """
    # Konwersja do skali szarości
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


def client_main(server_host, server_port=2040):
    """
    Główna funkcja klienta.
    """
    print(f"Łączenie z serwerem {server_host}:{server_port}...")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_host, server_port))

    print("Połączono! Odbieranie fragmentu...")
    fragment = receive_all(client_socket)

    print("Przetwarzanie fragmentu...")
    processed_fragment = edge_filter(fragment)

    print("Wysyłanie przetworzonego fragmentu...")
    send_all(client_socket, processed_fragment)

    client_socket.close()
    print("Fragment przetworzony i wysłany z powrotem do serwera")


if __name__ == "__main__":
    # Parametry do dostosowania
    SERVER_HOST = "10.182.30.192"  # IP serwera - zmień na właściwy
    SERVER_PORT = 2040

    client_main(SERVER_HOST, SERVER_PORT)