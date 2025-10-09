import cv2
import time 
import os
from .directory import directory
from .AI_face import learning

def sv():
    print("Нужно создать директорию dataset (yes/not)")
    create_dir=input(str())
    if create_dir=="yes":
            directory()
    else:
        
    # --- Создание основной папки directory ---
    # предполагаю, что эта функция создаёт папку "directory"
    
        # --- Запрос ФИО ---
        fio = input("Введите ФИО пользователя: ").strip()
    
        # --- Полный путь для сохранения фото ---
        save_dir = os.path.join("dataset", fio)
        os.makedirs(save_dir, exist_ok=True)  # создаём папку, если её нет
    
        # --- Настройка камеры ---
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
        if not cap.isOpened():
            print("Не удалось открыть камеру")
            exit()
    
        print("Камера запущена. Нажмите ENTER, чтобы начать съёмку...")
    
        # --- Ждем ENTER ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения кадра")
                break
            
            cv2.imshow("Camera", frame)
    
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                print("[INFO] Съёмка началась")
                break
            elif key == ord('q'):
                print("Выход без съёмки")
                cap.release()
                cv2.destroyAllWindows()
                exit()
    
        # --- Съёмка кадров ---
        for i in range(20):
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения кадра")
                break
            
            filename = os.path.join(save_dir, f"photo_{i+1}.png")
            cv2.imwrite(filename, frame)
    
            cv2.imshow("Camera", frame)
            time.sleep(0.5)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        print("Все фото сохранены в папке:", save_dir)
    
        print("Для обучения модели введите (yes/not)")
        checking=input(str())
        if checking=="yes":
            learning()
        else:
            return 0

