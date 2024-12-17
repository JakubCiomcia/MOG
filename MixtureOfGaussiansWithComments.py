import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry MOG
K = 5  # Liczba składników Gaussa w mieszance
alpha = 0.3  # Współczynnik uczenia (learning rate)
T = 0.7  # Próg wag tła (background weight threshold)


def rgb2gray(frame):
    """
    Konwertuje obraz z przestrzeni kolorów BGR (używanej przez OpenCV)
    do skali szarości.

    Args:
        frame (numpy.ndarray): Obraz w przestrzeni BGR.

    Returns:
        numpy.ndarray: Obraz w skali szarości jako typ float32.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


class MixtureOfGaussians:
    def __init__(self, height, width, K=3, init_var=15.0):
        """
        Inicjalizuje model Mieszanki Gaussów (Mixture of Gaussians - MOG) dla każdego piksela.

        Każdy piksel jest modelowany jako mieszanka K rozkładów Gaussa,
        z odpowiednimi wagami, średnimi i wariancjami.

        Args:
            height (int): Wysokość obrazu.
            width (int): Szerokość obrazu.
            K (int, optional): Liczba składników Gaussa w mieszance. Domyślnie 3.
            init_var (float, optional): Początkowa wariancja dla każdego Gaussa. Domyślnie 15.0.
        """
        self.height = height
        self.width = width
        self.K = K
        # Inicjalizacja wag wszystkich składników na równy podział (1/K)
        self.weights = np.ones((height, width, K)) * (1.0 / K)
        # Inicjalizacja średnich losowymi wartościami w zakresie [0, 255]
        self.means = np.random.rand(height, width, K) * 255.0
        # Inicjalizacja wariancji na wartość init_var
        self.vars = np.ones((height, width, K)) * init_var

    def update(self, frame_gray, alpha=0.01, T=0.7):
        """
        Aktualizuje model MOG na podstawie nowej klatki w skali szarości.

        Args:
            frame_gray (numpy.ndarray): Aktualna klatka w skali szarości.
            alpha (float, optional): Współczynnik uczenia. Domyślnie 0.01.
            T (float, optional): Próg wag tła. Domyślnie 0.7.

        Returns:
            numpy.ndarray: Maska pierwszoplanowych pikseli (foreground mask).
        """
        height, width = frame_gray.shape

        # Oblicza bezwzględną różnicę między bieżącą klatką a średnimi Gaussów
        diff = np.abs(self.means - frame_gray[:, :, None])  # Kształt: (height, width, K)

        # Sprawdza, które rozkłady Gaussa pasują do bieżącej obserwacji
        match = diff < 2.5 * np.sqrt(self.vars)  # Kształt: (height, width, K)

        # Dla każdego piksela znajduje indeks pierwszego dopasowanego składnika Gaussa
        matched_index = np.argmax(match, axis=2)  # Kształt: (height, width)
        # Maski wskazujące, które piksele mają przynajmniej jeden dopasowany składnik
        matched_mask = np.any(match, axis=2)  # Kształt: (height, width)

        # Tworzy indeksy wierszy i kolumn dla operacji wektorowych
        row_indices = np.arange(height)[:, None]  # Kształt: (height, 1)
        col_indices = np.arange(width)[None, :]  # Kształt: (1, width)

        # Aktualizuje wagi wszystkich składników, zmniejszając je o współczynnik uczenia
        self.weights *= (1 - alpha)

        # Dla dopasowanych składników zwiększa odpowiednie wagi
        gauss_idx = matched_index
        self.weights[row_indices, col_indices, gauss_idx] += alpha

        # Aktualizacja średnich i wariancji dopasowanych składników Gaussa
        rho = alpha
        matched_means = self.means[row_indices, col_indices, gauss_idx]
        matched_vars = self.vars[row_indices, col_indices, gauss_idx]
        x = frame_gray

        delta = x - matched_means
        self.means[row_indices, col_indices, gauss_idx] = matched_means + rho * delta
        self.vars[row_indices, col_indices, gauss_idx] = matched_vars + rho * (delta ** 2 - matched_vars)

        # Obsługa pikseli, które nie pasują do żadnego składnika Gaussa
        no_match_mask = ~matched_mask
        weakest_idx = np.argmin(self.weights, axis=2)

        # Pobranie współrzędnych pikseli bez dopasowania
        no_match_coords = np.where(no_match_mask)  # Tuple z tablicami współrzędnych (y, x)

        # Aktualizacja naj słabszego składnika Gaussa dla pikseli bez dopasowania
        self.weights[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = alpha
        self.means[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = frame_gray[no_match_mask]
        self.vars[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = 15.0

        # Normalizacja wag, aby suma wag dla każdego piksela wynosiła 1
        sum_w = np.sum(self.weights, axis=2)
        self.weights = self.weights / sum_w[:, :, None]

        # Wyznaczanie tła i pierwszoplanowych pikseli
        # Sortowanie składników Gaussa według wag malejąco
        sort_indices = np.argsort(self.weights, axis=2)[..., ::-1]
        weights_sorted = np.take_along_axis(self.weights, sort_indices, axis=2)
        means_sorted = np.take_along_axis(self.means, sort_indices, axis=2)
        vars_sorted = np.take_along_axis(self.vars, sort_indices, axis=2)

        # Oblicza skumulowane wagi dla określenia, które składniki należą do tła
        cum_weights = np.cumsum(weights_sorted, axis=2)
        fg_mask = np.ones((height, width), dtype=bool)  # Inicjalizuje maskę jako wszystkie piksele jako tło

        # Iteruje przez składniki Gaussa, aby wyznaczyć tło
        for i in range(self.K):
            bg_component_mask = cum_weights[:, :, i] < T  # Maski składników, które należą do tła
            original_index = sort_indices[:, :, i]  # Indeksy oryginalnych składników po sortowaniu
            # Oblicza odległość między średnią składnika a bieżącą klatką
            dist = np.abs(self.means[row_indices, col_indices, original_index] - frame_gray)
            # Ustala próg na podstawie wariancji
            threshold = 2.5 * np.sqrt(self.vars[row_indices, col_indices, original_index])
            # Sprawdza, czy odległość jest mniejsza niż próg
            match_bg = dist < threshold
            # Aktualizuje maskę pierwszoplanową, wykluczając piksele należące do tła
            fg_mask = fg_mask & ~(bg_component_mask & match_bg)

        # Zwraca maskę pierwszoplanowych pikseli jako obraz binarny (0 lub 255)
        return fg_mask.astype(np.uint8) * 255


def main():
    """
    Główna funkcja programu.

    Inicjalizuje przechwytywanie wideo, model MOG, tworzy okna do wyświetlania
    oryginalnej klatki i maski ruchu, oraz uruchamia animację.
    """
    # Inicjalizacja kamery za pomocą OpenCV
    cap = cv2.VideoCapture(0)  # Kamera domyślna (indeks 0)
    if not cap.isOpened():
        print("Nie można otworzyć kamerki.")
        return

    # Przechwytuje pierwszą klatkę, aby uzyskać rozmiary obrazu
    ret, frame = cap.read()
    if not ret:
        print("Nie można odczytać klatki z kamerki.")
        cap.release()
        return

    height, width = frame.shape[:2]  # Pobiera wysokość i szerokość obrazu
    mog = MixtureOfGaussians(height, width, K=K)  # Inicjalizuje model MOG

    # Tworzy dwa subplots: jeden do wyświetlania oryginalnej klatki, drugi do maski ruchu
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Oryginał")
    ax[1].set_title("Maska ruchu")

    def update(frame_number):
        """
        Funkcja aktualizująca animację. Czyta nową klatkę z kamery, aktualizuje model MOG,
        oraz wyświetla oryginalną klatkę i maskę ruchu.

        Args:
            frame_number (int): Numer aktualnej klatki (nieużywane).
        """
        ret, frame = cap.read()  # Czyta nową klatkę z kamery
        if not ret:
            return  # Jeśli nie udało się odczytać klatki, przerywa aktualizację

        frame_gray = rgb2gray(frame)  # Konwertuje klatkę do skali szarości
        fg_mask = mog.update(frame_gray, alpha=alpha, T=T)  # Aktualizuje model MOG i uzyskuje maskę ruchu

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konwertuje klatkę do przestrzeni RGB dla wyświetlania
        ax[0].imshow(frame_rgb)  # Wyświetla oryginalną klatkę
        ax[1].imshow(fg_mask, cmap='gray', vmin=0, vmax=255)  # Wyświetla maskę ruchu
        ax[0].axis('off')  # Ukrywa osie na pierwszym subplotcie
        ax[1].axis('off')  # Ukrywa osie na drugim subplotcie

    # Tworzy animację, która wywołuje funkcję update co 50 ms
    ani = FuncAnimation(fig, update, interval=50)
    plt.show()  # Wyświetla okno z animacją

    cap.release()  # Zwolnienie zasobów kamery po zakończeniu programu


if __name__ == "__main__":
    main()
