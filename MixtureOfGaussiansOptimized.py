import time  # Biblioteka do mierzenia czasu (funkcje time.time(), sleep itp.)
import cv2  # OpenCV - biblioteka do przetwarzania i analizy obrazów
import numpy as np  # NumPy - biblioteka do obliczeń numerycznych
import matplotlib.pyplot as plt  # Matplotlib - biblioteka do tworzenia wykresów
from matplotlib.animation import FuncAnimation  # FuncAnimation - umożliwia tworzenie animacji w Matplotlib

# Parametry MOG
K = 5  # liczba Gaussów (składników mieszanki) używanych w modelu
alpha = 0.05  # współczynnik uczenia (learning rate)
T = 0.7  # próg wag (threshold) decydujący, które Gaussy zaliczamy do tła
listaKlatek = []  # lista do przechowywania czasów przetwarzania kolejnych klatek


def rgb2gray(frame):
    # Konwersja obrazu z przestrzeni BGR (domyślnej w OpenCV) do skali szarości.
    # Zwraca tablicę z wartościami w formacie float32.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


class MixtureOfGaussians:
    def __init__(self, height, width, K=3, init_var=15.0):
        """
        Inicjalizacja klasy MOG.
        Dla każdego piksela przechowujemy mieszankę K rozkładów Gaussa:
        - weights[h, w, k]: wagi składnika k
        - means[h, w, k]: średnia składnika k
        - vars[h, w, k]: wariancja składnika k
        """
        self.height = height  # wysokość obrazu
        self.width = width  # szerokość obrazu
        self.K = K  # liczba składników Gaussa
        self.weights = np.ones((height, width, K)) * (1.0 / K)  # inicjalne wagi (równe)
        self.means = np.random.rand(height, width, K) * 255.0  # inicjalne średnie w zakresie 0-255
        self.vars = np.ones((height, width, K)) * init_var  # inicjalne wariancje (zadana stała)

    def update(self, frame_gray, alpha=0.05, T=0.7):
        # Pobranie wymiarów aktualnej klatki
        height, width = frame_gray.shape

        # Różnica między obserwacją (frame_gray) a średnimi (dla wszystkich K składników)
        # Rozszerzenie frame_gray do 3. wymiaru: frame_gray[:, :, None]
        diff = np.abs(self.means - frame_gray[:, :, None])  # shape: (h, w, K)

        # Sprawdzenie dopasowania do każdego z Gaussów:
        # Piksel jest uznawany za dopasowany do składnika, jeśli różnica jest mniejsza niż 2.5*sqrt(variancji)
        match = diff < 2.5 * np.sqrt(self.vars)  # zmienna bool o kształcie (h, w, K)

        # Dla każdego piksela znajduje indeks najbardziej pasującego składnika
        matched_index = np.argmax(match, axis=2)  # wybieramy składnik, dla którego match=True (o najwyższym arg)

        # Sprawdzamy, czy piksel pasuje do któregoś składnika
        matched_mask = np.any(match, axis=2)  # czy piksel pasuje do któregokolwiek z K składników

        # Przygotowujemy indeksy wierszy i kolumn (używane do indeksowania zaawansowanego)
        row_indices = np.arange(height)[:, None]  # indeksy wierszy
        col_indices = np.arange(width)[None, :]  # indeksy kolumn

        # Pomniejszamy wagi wszystkich składników o (1 - alpha)
        self.weights *= (1 - alpha)

        # Dla pikseli, które się dopasowały:
        # Zwiększamy wagę dopasowanego składnika o alpha
        gauss_idx = matched_index
        self.weights[row_indices, col_indices, gauss_idx] += alpha

        # Aktualizacja średnich i wariancji dopasowanego składnika:
        rho = alpha  # współczynnik aktualizacji
        matched_means = self.means[row_indices, col_indices, gauss_idx]  # średnie dopasowanego składnika
        matched_vars = self.vars[row_indices, col_indices, gauss_idx]  # wariancje dopasowanego składnika
        x = frame_gray  # aktualna obserwacja (klatka w skali szarości)

        # Zmiana średniej o część rho * (wartość pikseli - dotychczasowa średnia)
        delta = x - matched_means
        self.means[row_indices, col_indices, gauss_idx] = matched_means + rho * delta

        # Zmiana wariancji o część rho * (różnica kwadratowa - dotychczasowa wariancja)
        self.vars[row_indices, col_indices, gauss_idx] = matched_vars + rho * (delta ** 2 - matched_vars)

        # Obsługa pikseli, które nie dopasowały się do żadnego składnika
        no_match_mask = ~matched_mask  # negacja maski dopasowania

        # Znajdujemy indeks najsłabszego składnika (tego o najmniejszej wadze) dla każdego piksela
        weakest_idx = np.argmin(self.weights, axis=2)

        # Pobieramy współrzędne pikseli bez dopasowania
        no_match_coords = np.where(no_match_mask)

        # Dla tych pikseli aktualizujemy najsłabsze składniki, inicjalizując je jako "nowe"
        # (ustawiamy ich wagi na alpha, średnie na wartość obserwowaną, wariancje na 15.0)
        self.weights[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = alpha
        self.means[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = frame_gray[no_match_mask]
        self.vars[no_match_coords[0], no_match_coords[1], weakest_idx[no_match_mask]] = 15.0

        # Normalizacja wag – po aktualizacji wszystkie wagi muszą sumować się do 1
        sum_w = np.sum(self.weights, axis=2)
        self.weights = self.weights / sum_w[:, :, None]

        # Wyznaczenie tła i pierwszoplanowych pikseli:
        # Sortujemy składniki względem wag malejąco
        sort_indices = np.argsort(self.weights, axis=2)[...,
                       ::-1]  # indeksy po sortowaniu od największych do najmniejszych wag
        weights_sorted = np.take_along_axis(self.weights, sort_indices, axis=2)
        means_sorted = np.take_along_axis(self.means, sort_indices, axis=2)
        vars_sorted = np.take_along_axis(self.vars, sort_indices, axis=2)

        # Wyznaczamy skumulowane wagi
        cum_weights = np.cumsum(weights_sorted, axis=2)

        # Przygotowanie maski tła:
        # Tworzymy maskę, która wyznacza, które składniki wziąć pod uwagę (czyli dopóki skumulowana waga < T)
        bg_component_mask = cum_weights < T

        # Oryginalne indeksy składników (zostają, aby wiedzieć, który Gauss jest który)
        original_index = sort_indices  # tutaj przypisujemy posortowane indeksy do zmiennej

        # Obliczamy odległości i progi dla wszystkich składników
        dist = np.abs(means_sorted - frame_gray[:, :, None])  # odległość od obserwacji
        threshold = 2.5 * np.sqrt(vars_sorted)  # granica dopasowania (dla 2.5 * odchylenie std)

        # Tworzymy maskę dopasowania pikseli do składnika tła
        match_bg = dist < threshold

        # Łączymy maskę tła (bg_component_mask) z maską dopasowania (match_bg)
        bg_mask = bg_component_mask & match_bg

        # Sprawdzamy, czy piksel pasuje do któregoś składnika tła (any() wzdłuż osi K)
        bg_mask_sum = np.any(bg_mask, axis=2)

        # Maska pierwszoplanowa (fg) to negacja maski tła
        fg_mask = bg_mask_sum

        # Zwracamy maskę pierwszoplanową w postaci 0/255 (dla lepszej wizualizacji)
        return fg_mask.astype(np.uint8) * 255


def main():
    # Inicjalizacja kamery za pomocą OpenCV
    cap = cv2.VideoCapture(0)  # 0 – używamy domyślnej kamery
    if not cap.isOpened():
        # Jeśli kamera nie została otwarta, wyświetlamy komunikat i kończymy
        print("Nie można otworzyć kamerki.")
        return

    # Odczyt pierwszej klatki z kamery
    ret, frame = cap.read()
    if not ret:
        # Jeśli nie udało się odczytać klatki, zwalniamy kamerę i kończymy
        print("Nie można odczytać klatki z kamerki.")
        cap.release()
        return

    # Pobieramy rozmiar (wysokość, szerokość) pierwszej klatki
    height, width = frame.shape[:2]

    # Inicjalizujemy nasz model MOG z określoną liczbą składników K
    mog = MixtureOfGaussians(height, width, K=K)

    # Tworzymy figurę i dwie osie do wyświetlania: obraz oryginalny i maskę ruchu
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Oryginał")
    ax[1].set_title("Maska ruchu")

    def update(frame_number):
        # Funkcja wywoływana co klatkę przez FuncAnimation
        # Pobiera kolejną klatkę z kamery
        ret, frame = cap.read()
        if not ret:
            return

        # Konwersja aktualnej klatki do skali szarości
        frame_gray = rgb2gray(frame)

        # Mierzymy czas przed aktualizacją MOG
        t = time.time()

        # Aktualizujemy model MOG i otrzymujemy maskę pierwszoplanową
        fg_mask = mog.update(frame_gray, alpha=alpha, T=T)

        # Mierzymy czas po aktualizacji
        t2 = time.time()

        # Różnica (czas przetwarzania 1 klatki)
        t3 = t2 - t

        # Dodajemy zmierzony czas do listy
        listaKlatek.append(t3)

        # Wykonujemy obliczenia średniego czasu co 10 klatek
        suma = 0
        if (len(listaKlatek) % 10):
            for k in listaKlatek:
                suma += k

            srednia = suma / 10
            listaKlatek.clear()

            # Wyświetlamy średni czas w konsoli
            print(srednia)

        # Konwersja klatki do RGB (do poprawnego wyświetlania w Matplotlib)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wyświetlenie klatki oryginalnej
        ax[0].imshow(frame_rgb)
        # Wyświetlenie maski ruchu (fg_mask) w skali szarości
        ax[1].imshow(fg_mask, cmap='gray', vmin=0, vmax=255)

        # Ukrywamy osie (siatkę, wartości) dla ładniejszego widoku
        ax[0].axis('off')
        ax[1].axis('off')

    # Tworzymy animację, która co 50 ms wywołuje funkcję update
    ani = FuncAnimation(fig, update, interval=50)

    # Wyświetlamy okno Matplotlib z animacją
    plt.show()

    # Gdy okno zostanie zamknięte, zwalniamy zasoby kamery
    cap.release()


if __name__ == "__main__":
    main()
