import os
import sys
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np


class App:
    def __init__(self, root, window_title):
        self.root = root
        self.root.title(window_title)
        self.cap = cv2.VideoCapture(0)      # номер камери
        if not self.cap.isOpened():
            print("Camera not available")
            self.on_closing()
            exit(1)
        self.cap.set(3, 640)    # Ширина камери
        self.cap.set(4, 480)    # Висота камери

        # налаштування:
        self.min_size = 100             # мінімальний розмір числа в пікселях [0:]
        self.threshold = 110            # межа між 0 і 1. [0; 255]
        self.scaled_size = (40, 30)     # розмір вихідного зображення
        self.line_color = (0, 255, 0)   # колір рамки навколо об'єктів
        self.text_color = (255, 0, 0)   # колір рамки навколо об'єктів
        self.line_thickness = 1         # товщина рамки в пікселях
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # шрифт тексту на камері
        self.is_mu_max = False          # True - максимальне значення, False - мінімальне
        self.mu_threshold = 0.5         # мінімальний допустимий коефіцієнт

        self.frame_delay = 10  # ms
        self.BLACK = np.uint8(0)
        self.WHITE = np.uint8(255)
        self.EDGE = np.uint8(245)  # те саме що self.WHITE
        self.ONE = np.uint8(1)

        self.image = None
        self.one_bit_image = None
        self.obj_count = 0
        self.obj_bounds = np.zeros((self.EDGE, 4), dtype=np.uint16)
        self.object_mu = np.zeros((self.EDGE, 11), dtype=float)  # 10 для кожної цифри + індекс класифікованого
        self.standards = None

        # todo вибір камери
        # camera_info = cv2.getBuildInformation()
        # camera_list = [f'Камера {i}' for i in range(4)]
        # self.available_cameras = camera_list
        # self.camera_combobox = ttk.Combobox(window, values=self.available_cameras)
        # self.camera_combobox.set(self.available_cameras[0])
        # self.camera_combobox.pack()
        # self.start_button = tk.Button(window, text="Старт", command=self.start_camera)
        # self.start_button.pack()

        self.left_label = tk.Label(self.root, text="Зображення з камери:")
        self.left_label.grid(row=0, column=0)
        self.right_label = tk.Label(self.root, text="Машинне зображення:")
        self.right_label.grid(row=0, column=1)

        self.left_image = tk.Label(self.root)
        self.left_image.grid(row=1, column=0)
        self.right_image = tk.Label(self.root)
        self.right_image.grid(row=1, column=1)

        self.label_min_size = tk.Label(self.root, text="Мінімальний розмір об'єкта в пікселях:  [0:]")
        self.label_min_size.grid(row=2, column=0, columnspan=2)
        self.entry_min_size = tk.Entry(self.root)
        self.entry_min_size.grid(row=3, column=0, columnspan=2)
        self.entry_min_size.insert(0, str(self.min_size))

        self.label_threshold = tk.Label(self.root, text="Межа між 0 і 1:  [0:255]")
        self.label_threshold.grid(row=4, column=0, columnspan=2)
        self.entry_threshold = tk.Entry(self.root)
        self.entry_threshold.grid(row=5, column=0, columnspan=2)
        self.entry_threshold.insert(0, str(self.threshold))

        self.update_value_button = tk.Button(self.root, text="Оновити параметри", command=self.update_entry_value)
        self.update_value_button.grid(row=6, column=0, columnspan=2)
        self.save_button = tk.Button(self.root, text="Зберегти у файл", command=self.resize_and_save)
        self.save_button.grid(row=7, column=0, columnspan=2)

        self.is_classification = tk.IntVar()
        self.checkbox = tk.Checkbutton(self.root, text="Класифікувати", variable=self.is_classification)
        self.checkbox.grid(row=8, column=0, columnspan=2)

        self.load_standards()
        self.update_entry_value()
        # self.update_frame()  # no timer
        # self.update_frame_2()  # long timer
        self.update_frame_3()  # short timer

        # Функція для завершення програми при закритті вікна
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # завантажити кольорове зображення з камери
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Конвертувати кадр в чорно-білий
            self.binarize()
            # забрати чорне по краях
            self.highlight_edge_connected_region()
            # розпізнавати об'єкти
            self.find_objects()
            # знайти координати усіх прямокутників.
            self.find_rectangles_borders()
            # Намалювати прямокутник на кадрі
            self.draw_rectangles()
            # класифікація об'єктів
            if self.is_classification.get() == 1:
                self.classify()

            photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
            photo_bw = ImageTk.PhotoImage(image=Image.fromarray(self.one_bit_image))

            self.left_image.config(image=photo)
            self.left_image.image = photo
            self.right_image.config(image=photo_bw)
            self.right_image.image = photo_bw

        self.root.after(self.frame_delay, self.update_frame)

    def update_frame_2(self):  # timer for each step. todo remove
        ret, frame = self.cap.read()
        if ret:
            start_time = time.time()
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 1 color: ", execution_time * 1000)
            start_time = time.time()
            self.binarize()
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 2 black: ", execution_time * 1000)
            start_time = time.time()
            self.highlight_edge_connected_region()
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 3 edges: ", execution_time * 1000)
            start_time = time.time()
            self.find_objects()
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 4 objct: ", execution_time * 1000)
            start_time = time.time()
            self.find_rectangles_borders()
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 5 bords: ", execution_time * 1000)
            start_time = time.time()
            self.draw_rectangles()
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 6 draws: ", execution_time * 1000)
            start_time = time.time()
            if self.is_classification.get() == 1:
                self.classify()
            end_time = time.time()
            execution_time = end_time - start_time
            print("Time 7 class: ", execution_time * 1000)
            print('.')

            photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
            photo_bw = ImageTk.PhotoImage(image=Image.fromarray(self.one_bit_image))

            self.left_image.config(image=photo)
            self.left_image.image = photo
            self.right_image.config(image=photo_bw)
            self.right_image.image = photo_bw

        self.root.after(self.frame_delay, self.update_frame_2)

    def update_frame_3(self):  # timer for each frame. todo remove
        ret, frame = self.cap.read()
        if ret:
            start_time = time.time()

            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.binarize()
            self.highlight_edge_connected_region()
            self.find_objects()
            self.find_rectangles_borders()
            self.draw_rectangles()
            if self.is_classification.get() == 1:
                self.classify()

            end_time = time.time()
            execution_time = end_time - start_time
            print("Total time: ", execution_time * 1000)

            photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
            photo_bw = ImageTk.PhotoImage(image=Image.fromarray(self.one_bit_image))

            self.left_image.config(image=photo)
            self.left_image.image = photo
            self.right_image.config(image=photo_bw)
            self.right_image.image = photo_bw

        self.root.after(self.frame_delay, self.update_frame_3)

    def binarize(self):  # робить зображення чорно-білим, межа по self.threshold
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.one_bit_image = np.where(gray_image > self.threshold, self.WHITE, self.BLACK)

    def highlight_edge_connected_region(self):  # забирає чорні об'єкти які торкаються країв todo speed up
        # Get image dimensions
        width, height = self.one_bit_image.shape

        # Function for flood-filling a connected black region
        def boundary_fill(x, y):
            stack = [(x, y)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= width or y < 0 or y >= height or self.one_bit_image[x, y] != self.BLACK:
                    continue
                self.one_bit_image[x, y] = self.EDGE
                stack.append((x - 1, y))
                stack.append((x + 1, y))
                stack.append((x, y - 1))
                stack.append((x, y + 1))

        # Process all edges in one loop
        for i in range(width):
            boundary_fill(i, 0)
            boundary_fill(i, height - 1)

        for j in range(height):
            boundary_fill(0, j)
            boundary_fill(width - 1, j)

    def find_objects(self):  # розділяє чорні об'єкти (0) на різні об'єкти (1,2,3,4...) todo speed up too
        width, height = self.one_bit_image.shape

        def fill_objects(x, y, prev_color, fill_color):
            stack = [(x, y)]
            count = 0
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= width or y < 0 or y >= height or self.one_bit_image[x, y] != prev_color:
                    continue
                self.one_bit_image[x, y] = fill_color
                count += 1
                stack.append((x - 1, y))
                stack.append((x + 1, y))
                stack.append((x, y - 1))
                stack.append((x, y + 1))
            return count

        color = np.uint8(1)  # чорний, але трішки інший для кожного об'єкту
        for i in range(1, width - 1):  # крайні лінії гарантовано не чорні, тому 1 і -1
            for j in range(1, height - 1):
                if self.one_bit_image[i, j] < self.EDGE:  # якщо це не фон
                    size = fill_objects(i, j, self.BLACK, color)  # виділяємо об'єкт
                    if size > self.min_size:  # якщо об'єкт великий, то переходимо до наступного
                        color += self.ONE
                        if color > self.EDGE:  # якщо не вистачить чисел для об'єктів, то потрібно збільшити self.min_n
                            print(f"Забагато об'єктів ({color})! Радимо збільшити мінімальний розмір об'єкта.")
                            return
                    else:
                        fill_objects(i, j, color, self.EDGE)  # малі об'єкти перемальовуємо в білий
        self.obj_count = color  # кількість об'єктів в кадрі

    def find_rectangles_borders(self):  # знаходимо межі для кожного об'єкту
        self.obj_bounds = np.zeros_like(self.obj_bounds)  # скидуємо значення з попереднього кадру
        self.obj_bounds[:, [0, 1]] -= self.ONE

        for k in range(1, self.obj_count):
            obj_k_pixels = (self.one_bit_image == k).nonzero()
            if len(obj_k_pixels[0]) > 0:
                obj_k_min_x = np.min(obj_k_pixels[0])
                obj_k_max_x = np.max(obj_k_pixels[0])
                obj_k_min_y = np.min(obj_k_pixels[1])
                obj_k_max_y = np.max(obj_k_pixels[1])
                self.obj_bounds[k, 0] = obj_k_min_x  # якщо це крайнє верхнє
                self.obj_bounds[k, 2] = obj_k_max_x  # якщо це крайнє нижнє
                self.obj_bounds[k, 1] = obj_k_min_y  # якщо це крайнє ліве
                self.obj_bounds[k, 3] = obj_k_max_y  # якщо це крайнє праве

    def draw_rectangles(self):  # малюємо усі прямокутники
        for k in range(1, self.obj_count):
            x1 = int(self.obj_bounds[k, 0])
            y1 = int(self.obj_bounds[k, 1])
            x2 = int(self.obj_bounds[k, 2])
            y2 = int(self.obj_bounds[k, 3])
            cv2.rectangle(self.image, (y1, x1), (y2, x2), self.line_color, self.line_thickness)
            # cv2.rectangle(self.one_bit_image, (y1, x1), (y2, x2), self.line_color, self.line_thickness)

    def resize_obj(self, k):
        x1 = int(self.obj_bounds[k, 0])
        y1 = int(self.obj_bounds[k, 1])
        x2 = int(self.obj_bounds[k, 2])
        y2 = int(self.obj_bounds[k, 3])
        cropped_image = self.one_bit_image[x1:x2, y1:y2].copy()  # обрізаємо зайве
        cropped_image[cropped_image == k] = self.BLACK  # робимо чорно-білим
        cropped_image[cropped_image != self.BLACK] = self.WHITE
        resized_image = cv2.resize(cropped_image, self.scaled_size)  # змінюємо розмір
        return resized_image

    def resize_and_save(self):
        for rt, dirs, files in os.walk("scaled_objects"):  # видаляємо попередні об'єкти із папки
            for file in files:
                file_path = os.path.join(rt, file)
                os.remove(file_path)
        for k in range(1, self.obj_count):
            resized_image = self.resize_obj(k)
            cv2.imwrite(f"scaled_objects/png/object_№{k}.png", resized_image)  # зберігаємо у файл 
            np.savetxt(f"scaled_objects/txt/object_№{k}.txt", resized_image, delimiter='\t', fmt='%d')
            self.classify()  # todo remove second resize_obj
            np.savetxt(f"scaled_objects/mu_object_№{k}=={int(self.object_mu[k, -1])}.txt",
                       self.object_mu[k], fmt='%.4g')

    def classify(self):
        for k in range(1, self.obj_count):
            resized_image = self.resize_obj(k)
            for d in range(10):
                self.object_mu[k, d] = self.classify_func(d, resized_image)

            if self.is_mu_max:  # mu --> max
                self.object_mu[k, -1] = sys.float_info.min
                index = np.argmax(self.object_mu[k])
            else:  # mu --> min
                self.object_mu[k, -1] = sys.float_info.max
                index = np.argmin(self.object_mu[k])
            mu = self.object_mu[k, index]
            self.object_mu[k, -1] = index  # цифра яка найбільше підходить

            if (self.is_mu_max and mu > self.mu_threshold) or (not self.is_mu_max and mu < self.mu_threshold):
                # text = f'{index}={mu:.2f}'
                text = str(index)
            else:
                # text = f'?={mu:.2f}'
                text = '?'

            x1 = int(self.obj_bounds[k, 0])
            y1 = int(self.obj_bounds[k, 1])
            org = y1, x1 + 13  # зміщення вниз
            font_scale = 0.5
            self.image = cv2.putText(self.image, text, org, self.font, font_scale,
                                     self.text_color, self.line_thickness, cv2.LINE_AA)

    def classify_func(self, d, tested_object):
        standard = self.standards[d]
        if standard.shape != tested_object.shape:
            print(standard.shape)
            print(tested_object.shape)
            raise ValueError("Розміри матриць не збігаються")

        # todo змінити варіант, max/min, mu_threshold
        return np.sum(tested_object * standard) / np.sum(tested_object * tested_object)  # 3 варіант

    def load_standards(self):
        self.standards = []
        for d in range(10):
            s = np.loadtxt(f"standards/standard_{d}.txt", dtype=np.uint8)
            self.standards.append(s)

    def update_entry_value(self):
        min_size = int(self.entry_min_size.get())
        threshold = int(self.entry_threshold.get())
        if 0 <= min_size <= 5000:
            self.min_size = min_size
        if 0 <= threshold <= 255:
            self.threshold = threshold

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == "__main__":
    # print_hi('PyCharm')
    root = tk.Tk()
    app = App(root, "Борак Дмитро КН-404")
    root.mainloop()
