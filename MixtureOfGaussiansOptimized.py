import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry MOG
K = 5  # liczba Gaussów
alpha = 0.05  # współczynnik uczenia
T = 0.7  # próg wag tła
listaKlatek = []


def rgb2gray(frame):
    # Konwersja obrazu BGR (OpenCV) do skali szarości
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


class MixtureOfGaussians:
    def __init__(self, height, width, K=3, init_var=15.0):
        """
        Inicjalizacja MOG.
        Każdy piksel to mieszanka K rozkładów Gaussa:
        weights[h, w, k], means[h, w, k], vars[h, w, k]
        """
        self.height = height
        self.width = width
        self.K = K
        self.weights = np.ones((height, width, K)) * (1.0 / K)
        self.means = np.random.rand(height, width, K) * 255.0
        self.vars = np.ones((height, width, K)) * init_var

    def update(self, frame_gray, alpha=0.05, T=0.7):
        height, width = frame_gray.shape
        # Różnica między aktualną obserwacją a średnimi
        diff = np.abs(self.means - frame_gray[:, :, None])  # (h,w,K)
        # Sprawdź dopasowanie do któregoś z Gaussów
        match = diff < 2.5 * np.sqrt(self.vars)  # bool (h,w,K)

        # Indeks dopasowanego składnika
        matched_index = np.argmax(match, axis=2)
        matched_mask = np.any(match, axis=2)

        row_indices = np.arange(height)[:, None]
        col_indices = np.arange(width)[None, :]

        # Aktualizacja wag (wszystkie wagi pomniejszamy)
        self.weights *= (1 - alpha)

        # Dla dopasowanych:
        gauss_idx = matched_index
        self.weights[row_indices, col_indices, gauss_idx] += alpha

        # Aktualizacja średnich i wariancji dopasowanego składnika:
        rho = alpha
        matched_means = self.means[row_indices, col_indices, gauss_idx]
        matched_vars = self.vars[row_indices, col_indices, gauss_idx]
        x = frame_gray

        delta = x - matched_means
        self.means[row_indices, col_indices, gauss_idx] = matched_means + rho * delta
        self.vars[row_indices, col_indices, gauss_idx] = matched_vars + rho * (delta ** 2 - matched_vars)

        # Obsługa pikseli bez dopasowania:
        no_match_mask = ~matched_mask
        weakest_idx = np.argmin(self.weights, axis=2)

        # Zmiana: użycie np.where do uzyskania współrzędnych pikseli bez dopasowania
        no_match_coords = np.where(no_match_mask)  # (rows, cols)

        # Aktualizacja najsłabszych składników dla pikseli bez dopasowania
        self.weights[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = alpha
        self.means[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = frame_gray[no_match_mask]
        self.vars[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = 15.0

        # Normalizacja wag
        sum_w = np.sum(self.weights, axis=2)
        self.weights = self.weights / sum_w[:, :, None]

        # Wyznaczenie tła i pierwszoplanowych pikseli:
        # Sortowanie składników wg wag malejąco
        sort_indices = np.argsort(self.weights, axis=2)[..., ::-1]
        weights_sorted = np.take_along_axis(self.weights, sort_indices, axis=2)
        means_sorted = np.take_along_axis(self.means, sort_indices, axis=2)
        vars_sorted = np.take_along_axis(self.vars, sort_indices, axis=2)

        cum_weights = np.cumsum(weights_sorted, axis=2)

        # Przygotowanie do maski ruchu
        # Zamiast pętli, użyjemy wektorowych operacji
        # Tworzymy maskę, która będzie oznaczać tło
        bg_component_mask = cum_weights < T  # (h,w,K)

        # Pobieramy oryginalne indeksy składników po sortowaniu
        # Indeksy są już posortowane, więc możemy użyć zaawansowanego indeksowania
        original_index = sort_indices  # (h,w,K)

        # Obliczamy odległości i progi dla wszystkich składników jednocześnie
        dist = np.abs(means_sorted - frame_gray[:, :, None])  # (h,w,K)
        threshold = 2.5 * np.sqrt(vars_sorted)  # (h,w,K)

        # Tworzymy maskę dopasowania do tła
        match_bg = dist < threshold  # (h,w,K)

        # Kombinujemy maskę tła i dopasowanie
        bg_mask = bg_component_mask & match_bg  # (h,w,K)

        # Sumujemy po trzecim wymiarze, aby znaleźć, czy którykolwiek składnik pasuje do tła
        bg_mask_sum = np.any(bg_mask, axis=2)  # (h,w)

        # Maskujemy maskę ruchu
        fg_mask = ~bg_mask_sum  # (h,w)

        return fg_mask.astype(np.uint8) * 255


def main():
    # Inicjalizacja kamery za pomocą OpenCV
    cap = cv2.VideoCapture(0)  # kamera domyślna
    if not cap.isOpened():
        print("Nie można otworzyć kamerki.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Nie można odczytać klatki z kamerki.")
        cap.release()
        return

    height, width = frame.shape[:2]
    mog = MixtureOfGaussians(height, width, K=K)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Oryginał")
    ax[1].set_title("Maska ruchu")

    def update(frame_number):
        ret, frame = cap.read()
        if not ret:
            return
        frame_gray = rgb2gray(frame)

        t = time.time()

        fg_mask = mog.update(frame_gray, alpha=alpha, T=T)

        t2 = time.time()

        t3 = t2 - t

        listaKlatek.append(t3)

        suma = 0
        if (len(listaKlatek) % 10):
            for k in listaKlatek:
                suma += k

            srednia =  suma / 10
            listaKlatek.clear()

            print(srednia)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax[0].imshow(frame_rgb)
        ax[1].imshow(fg_mask, cmap='gray', vmin=0, vmax=255)
        ax[0].axis('off')
        ax[1].axis('off')

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()

    cap.release()


if __name__ == "__main__":
    main()
