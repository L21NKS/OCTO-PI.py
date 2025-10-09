import cv2
import time
import os
import datetime
from motion_detection import detect_motion, draw_motion_visualization
from camera_utils import (
    initialize_cameras, release_cameras, create_video_grid,
    get_no_signal_frame, get_waiting_frame,
    MultiMaskCreator, load_mask, overlay_mask,load_lbph_face_recognizer
)
from view_logs import view_logs
from AI_face_detection.script_save import sv
from camera_utils import detect_and_recognize_faces
from logger import motion_logger




print(r"""________  ____________________________        /\ __________.___                   
\_____  \ \_   ___ \__    ___/\_____  \      / / \______   \   |    ______ ___.__.
 /   |   \/    \  \/ |    |    /   |   \    / /   |     ___/   |    \____ <   |  |
/    |    \     \____|    |   /    |    \  / /    |    |   |   |    |  |_> >___  |
\_______  /\______  /|____|   \_______  / / /     |____|   |___| /\ |   __// ____|
        \/        \/                  \/  \/                     \/ |__|   \/     """)




class SurveillanceSystem:
    def __init__(self):
        self.camera_indices = [0, 1, 2, 3]
        self.caps = []
        self.recognizer = None
        self.label_dict = None
        self.face_cascade = None
        self.masks = {}  # {camera_idx: mask}

        # Состояние камер
        self.motion_detected = {idx: False for idx in self.camera_indices}
        self.prev_frames = {idx: None for idx in self.camera_indices}
        self.last_motion_time = {idx: 0 for idx in self.camera_indices}
        self.last_motion_check = {idx: 0 for idx in self.camera_indices}
        self.motion_start_time = {idx: 0 for idx in self.camera_indices}
        self.motion_contours = {idx: [] for idx in self.camera_indices}
        self.last_check_time = {idx: 0 for idx in self.camera_indices}

        self.camera_triggered = []
        self.camera_faces = []
        self.camera_motion = []
        self.MOTION_TIMEOUT = 10
        self.CHECK_INTERVAL = 1
        self.MOTION_THRESHOLD = 25
        self.MOTION_MIN_AREA = 500

        self.active_motion_cameras = set()
        self.mask_creator = MultiMaskCreator()

    def main_menu(self):
        while True:
            print("\nГлавное меню")
            print("1. Запустить систему видеонаблюдения")
            print("2. Просмотреть логи")
            print("3. Создать биометрическую маску пользователя")
            print("4. Настройка масок")
            print("5. Выйти")

            choice = input("  ")

            if choice == "1":
                self.run()  # запуск системы
            elif choice == "2":
                view_logs()  # просмотр логов
            elif choice == "3":
                sv()
            elif choice == "4":
                self.setup_masks()
            elif choice == "5":
                print("[SYSTEM] Завершение работы")
                break
            else:
                print("\033[93mНеверный выбор\033[0m")

        
    def initialize(self):
        """Инициализация системы"""
        motion_logger.log_system_event("Инициализация системы видеонаблюдения")
        # Загрузка модели детектирования лиц
        try:
            self.recognizer, self.label_dict, self.face_cascade = load_lbph_face_recognizer(
                model_path="face_model.yml", 
                labels_path="labels.npy"
            )
            motion_logger.log_system_event("LBPH Face Recognizer загружен")
        except Exception as e:
            motion_logger.log_system_event(f"Ошибка загрузки LBPH модели: {e}")
            self.recognizer = None
            self.label_dict = None
            self.face_cascade = None


        # Инициализация камер
        self.caps = initialize_cameras(self.camera_indices)

        # Загрузка масок
        self.load_all_masks()

        # Получение настроек
        self.get_user_settings()

        motion_logger.log_system_event("Система инициализирована")

    def load_all_masks(self):
        """Загрузка всех масок из папки masks"""
        masks_dir = "masks"
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)
            return

        for filename in os.listdir(masks_dir):
            if filename.startswith("camera_") and filename.endswith(".png"):
                try:
                    parts = filename.split('_')
                    camera_idx = int(parts[1])
                    mask_path = os.path.join(masks_dir, filename)
                    mask = load_mask(mask_path)
                    if mask is not None:
                        self.masks[camera_idx] = mask
                        motion_logger.log_system_event(f"Загружена маска для камеры {camera_idx}")
                except (ValueError, IndexError):
                    continue

    def get_user_settings(self):
        """Получение настроек от пользователя"""
        print("\n\033[96mНастройка системы\033[0m")
        print("=" * 50)

        print("Введите номера камер для детектирования лиц:")
        print("Доступные камеры:", self.camera_indices)
        try:
            self.camera_faces = list(map(int, input("  ").split()))
        except Exception:
            self.camera_faces = []

        print("\nВведите номера камер для детектирования движения:")
        print("Доступные камеры:", self.camera_indices)
        try:
            self.camera_motion = list(map(int, input("  ").split()))
        except Exception:
            self.camera_motion = []

        print("\nВведите номера камер, которые только включаются по движению:")
        print("Доступные камеры:", self.camera_indices)
        try:
            self.camera_triggered = list(map(int, input("  ").split()))
        except Exception:
            self.camera_triggered = []

        print("\nВведите время таймаута после движения:")
        try:
            self.MOTION_TIMEOUT = int(input("  "))
        except Exception:
            pass

        # Настройка масок

        # Логирование настроек
        settings = {
            'cameras_faces': self.camera_faces,
            'cameras_motion': self.camera_motion,
            'cameras_triggered': self.camera_triggered,
            'timeout': self.MOTION_TIMEOUT,
            'threshold': self.MOTION_THRESHOLD,
            'min_area': self.MOTION_MIN_AREA,
            'masks': list(self.masks.keys())
        }
        motion_logger.log_settings(settings)

        # Статус камер
        for cam_idx in self.camera_indices:
            status_parts = []
            if cam_idx in self.camera_faces:
                status_parts.append("Детектирование лиц")
            if cam_idx in self.camera_motion:
                status_parts.append("Детектирование движения")
            if cam_idx in self.camera_triggered:
                status_parts.append("Только включение по движению")
            if cam_idx in self.masks:
                status_parts.append("Маска активна")
            status = " + ".join(status_parts) if status_parts else "Обычный режим"
            motion_logger.log_camera_status(cam_idx, status)

    def process_triggered_camera(self, camera_idx, frame, current_time):
        mask = self.masks.get(camera_idx)
        display_frame = frame.copy()

        if self.motion_detected[camera_idx]:
            # Активный режим
            if current_time - self.last_motion_check.get(camera_idx, 0) > 0.5:
                if self.prev_frames[camera_idx] is not None:
                    motion, contours = detect_motion(
                        self.prev_frames[camera_idx], frame,
                        self.MOTION_THRESHOLD, self.MOTION_MIN_AREA, mask
                    )
                    if motion:
                        self.last_motion_time[camera_idx] = current_time
                        motion_logger.log_system_event(f"Cam{camera_idx}: Движение продолжается")
                self.last_motion_check[camera_idx] = current_time
                self.prev_frames[camera_idx] = frame.copy()

            time_since_last_motion = current_time - self.last_motion_time[camera_idx]
            time_left = int(self.MOTION_TIMEOUT - time_since_last_motion)
            if time_since_last_motion > self.MOTION_TIMEOUT:
                if camera_idx in self.active_motion_cameras:
                    duration = current_time - self.motion_start_time[camera_idx]
                    motion_logger.log_motion_stopped(camera_idx, duration, 0)
                    self.active_motion_cameras.remove(camera_idx)
                self.motion_detected[camera_idx] = False
                self.last_check_time[camera_idx] = current_time
                motion_logger.log_camera_status(camera_idx, "Переход в режим ожидания")
                return get_waiting_frame(camera_idx)

            display_frame = draw_motion_visualization(frame, [], camera_idx, mask, time_left)

            # Face recognition
            if camera_idx in self.camera_faces and self.recognizer:
                display_frame, face_boxes = detect_and_recognize_faces(self.recognizer, self.label_dict, self.face_cascade, display_frame)
                if face_boxes:
                    cv2.putText(display_frame, f"Faces: {len(face_boxes)}", (15, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return display_frame

        else:
            # Режим ожидания
            time_since_last_check = current_time - self.last_check_time[camera_idx]
            if time_since_last_check >= self.CHECK_INTERVAL:
                if self.prev_frames[camera_idx] is not None:
                    motion, _ = detect_motion(
                        self.prev_frames[camera_idx], frame,
                        self.MOTION_THRESHOLD, self.MOTION_MIN_AREA, mask
                    )
                    if motion:
                        self.motion_detected[camera_idx] = True
                        self.last_motion_time[camera_idx] = current_time
                        self.motion_start_time[camera_idx] = current_time
                        self.last_motion_check[camera_idx] = current_time
                        motion_logger.log_motion_detected(camera_idx, is_triggered=True)
                        self.active_motion_cameras.add(camera_idx)
                        self.prev_frames[camera_idx] = frame.copy()
                        display_frame = draw_motion_visualization(frame, [], camera_idx, mask, self.MOTION_TIMEOUT)
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        return display_frame
                self.last_check_time[camera_idx] = current_time
                self.prev_frames[camera_idx] = frame.copy()

            waiting_frame = get_waiting_frame(camera_idx)
            if mask is not None:
                waiting_frame = overlay_mask(waiting_frame, mask)

            # Timestamp на кадре ожидания
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(waiting_frame, timestamp, (10, waiting_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return waiting_frame


    def process_motion_camera(self, camera_idx, frame, current_time):
        mask = self.masks.get(camera_idx)
        display_frame = frame.copy()
    
        if self.motion_detected[camera_idx]:
            motion, contours = detect_motion(
                self.prev_frames[camera_idx], frame,
                self.MOTION_THRESHOLD, self.MOTION_MIN_AREA, mask
            )
            if motion:
                self.last_motion_time[camera_idx] = current_time
                self.motion_contours[camera_idx] = contours
                objects_info = motion_logger.track_objects(camera_idx, contours)
                if objects_info['new_objects']:
                    motion_logger.log_new_objects(camera_idx, objects_info)
                motion_logger.log_motion_summary(camera_idx, objects_info)
    
            time_since_last_motion = current_time - self.last_motion_time[camera_idx]
            time_left = int(self.MOTION_TIMEOUT - time_since_last_motion)
            if time_since_last_motion > self.MOTION_TIMEOUT:
                if camera_idx in self.active_motion_cameras:
                    duration = current_time - self.motion_start_time[camera_idx]
                    total_objects = motion_logger.object_counter.get(camera_idx, 0)
                    motion_logger.log_motion_stopped(camera_idx, duration, total_objects)
                    self.active_motion_cameras.remove(camera_idx)
                self.motion_detected[camera_idx] = False
                self.motion_contours[camera_idx] = []
                self.last_check_time[camera_idx] = current_time
                motion_logger.log_camera_status(camera_idx, "Переход в режим ожидания")
                return get_waiting_frame(camera_idx)
    
            self.prev_frames[camera_idx] = frame.copy()
            display_frame = draw_motion_visualization(frame, self.motion_contours[camera_idx], camera_idx, mask, time_left)
    
            # Face recognition
            if camera_idx in self.camera_faces and self.recognizer:
                display_frame, face_boxes = detect_and_recognize_faces(self.recognizer, self.label_dict, self.face_cascade, display_frame)
                if face_boxes:
                    cv2.putText(display_frame, f"Faces: {len(face_boxes)}", (15, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            # Timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
            return display_frame

        else:
            # Камера ждёт движения
            time_since_last_check = current_time - self.last_check_time[camera_idx]
            if time_since_last_check >= self.CHECK_INTERVAL:
                if self.prev_frames[camera_idx] is not None:
                    motion, contours = detect_motion(
                        self.prev_frames[camera_idx], frame,
                        self.MOTION_THRESHOLD, self.MOTION_MIN_AREA, mask
                    )
                    if motion:
                        self.motion_detected[camera_idx] = True
                        self.last_motion_time[camera_idx] = current_time
                        self.motion_start_time[camera_idx] = current_time
                        self.motion_contours[camera_idx] = contours
                        objects_info = motion_logger.track_objects(camera_idx, contours)
                        if objects_info['new_objects']:
                            motion_logger.log_new_objects(camera_idx, objects_info)
                        motion_logger.log_motion_detected(camera_idx)
                        motion_logger.log_motion_summary(camera_idx, objects_info)
                        self.active_motion_cameras.add(camera_idx)
                        self.prev_frames[camera_idx] = frame.copy()
                        display_frame = draw_motion_visualization(frame, contours, camera_idx, mask, self.MOTION_TIMEOUT)
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        return display_frame

                self.last_check_time[camera_idx] = current_time
                self.prev_frames[camera_idx] = frame.copy()

            waiting_frame = get_waiting_frame(camera_idx)
            if mask is not None:
                waiting_frame = overlay_mask(waiting_frame, mask)

            # Timestamp на кадре ожидания
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(waiting_frame, timestamp, (10, waiting_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return waiting_frame


    def process_static_camera(self, camera_idx, frame):
        display_frame = frame.copy()
        mask = self.masks.get(camera_idx)
        if mask is not None:
            display_frame = overlay_mask(display_frame, mask)

        if camera_idx in self.camera_faces and self.recognizer is not None:
            display_frame, face_boxes = detect_and_recognize_faces(
                self.recognizer, self.label_dict, self.face_cascade, display_frame
            )
            if face_boxes:
                cv2.putText(display_frame, f"Faces: {len(face_boxes)}",
                            (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Добавляем timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(display_frame, timestamp, 
                    (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return display_frame


    def process_camera_frame(self, camera_idx, frame, current_time):
        if frame is None:
            return get_no_signal_frame(camera_idx)

        frame = cv2.resize(frame, (640, 480))

        # ✅ Камера одновременно в режимах TRIGGERED и MOTION
        if camera_idx in self.camera_triggered and camera_idx in self.camera_motion:
            if self.motion_detected[camera_idx]:
                # Камера уже активирована → работаем как motion-камера
                return self.process_motion_camera(camera_idx, frame, current_time)
            else:
                # Камера ждёт движения → используем поведение triggered
                return self.process_triggered_camera(camera_idx, frame, current_time)

        # Только TRIGGERED
        elif camera_idx in self.camera_triggered:
            return self.process_triggered_camera(camera_idx, frame, current_time)

        # Только MOTION
        elif camera_idx in self.camera_motion:
            return self.process_motion_camera(camera_idx, frame, current_time)

        # Статическая (обычный режим)
        else:
            return self.process_static_camera(camera_idx, frame)


    def setup_masks(self):
        """Создание масок"""
        print("\n\033[96mНастройка масок\033[0m")
        for cam_idx in self.camera_indices:
            print(f"\nСоздать маску для камеры {cam_idx} (y/n):")
            if input("  ").lower() == 'y':
                print("Введите имя маски (Enter = 'default'):")
                mask_name = input("  ").strip() or "default"
                mask_path = self.mask_creator.create_mask(cam_idx, mask_name)
                if mask_path:
                    mask = load_mask(mask_path)
                    if mask is not None:
                        self.masks[cam_idx] = mask
                        motion_logger.log_system_event(f"\033[92mСоздана маска для камеры {cam_idx}\033[0m")

    def run(self):
        try:
            self.initialize()
            while True:
                frames = []
                current_time = time.time()
                for idx, cap in enumerate(self.caps):
                    camera_idx = self.camera_indices[idx]
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            processed_frame = self.process_camera_frame(camera_idx, frame, current_time)
                        else:
                            processed_frame = get_no_signal_frame(camera_idx)
                            motion_logger.log_camera_status(camera_idx, "Нет сигнала")
                    else:
                        processed_frame = get_no_signal_frame(camera_idx)
                        motion_logger.log_camera_status(camera_idx, "Не найдена")
                    frames.append(cv2.resize(processed_frame, (320, 240)))

                grid = create_video_grid(frames, (2, 2), (640, 480))
                self.add_status_info(grid, current_time)
                cv2.imshow("Multi-Camera Surveillance System", grid)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_motion_cameras()
                elif key == ord('+'):
                    self.adjust_sensitivity(-5)
                elif key == ord('-'):
                    self.adjust_sensitivity(5)
                elif key == ord('m'):
                    self.setup_masks()
        finally:
            self.cleanup()

    def add_status_info(self, grid, current_time):
        status_lines = []
        for cam_idx in self.camera_indices:
            if cam_idx in self.camera_triggered:
                if self.motion_detected[cam_idx]:
                    time_left = int(self.MOTION_TIMEOUT - (current_time - self.last_motion_time[cam_idx]))
                    status = f"TRIGGERED ({time_left}s)"
                else:
                    next_check = int(self.CHECK_INTERVAL - (current_time - self.last_check_time[cam_idx]))
                    status = f"STANDBY ({next_check}s)"
            elif cam_idx in self.camera_motion:
                if self.motion_detected[cam_idx]:
                    time_left = int(self.MOTION_TIMEOUT - (current_time - self.last_motion_time[cam_idx]))
                    status = f"ACTIVE ({time_left}s)"
                else:
                    next_check = int(self.CHECK_INTERVAL - (current_time - self.last_check_time[cam_idx]))
                    status = f"STANDBY ({next_check}s)"
            else:
                status = "ALWAYS ON"

            if cam_idx in self.masks:
                status += " [MASK]"
            status_lines.append(f"Cam{cam_idx}:{status}")

        cv2.putText(grid, " | ".join(status_lines), (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        controls = "'r': reset | '+/-': sensitivity | 'm': masks | 'q': quit"
        cv2.putText(grid, controls, (10, grid.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def reset_motion_cameras(self):
        """Сброс состояния камер с детектированием движения"""
        for cam_idx in list(set(self.camera_motion + self.camera_triggered)):
            if cam_idx in self.active_motion_cameras:
                duration = time.time() - self.motion_start_time[cam_idx]
                total_objects = motion_logger.object_counter.get(cam_idx, 0)
                motion_logger.log_motion_stopped(cam_idx, duration, total_objects)
                self.active_motion_cameras.discard(cam_idx)

            self.motion_detected[cam_idx] = False
            self.prev_frames[cam_idx] = None
            self.last_motion_time[cam_idx] = 0
            self.last_motion_check[cam_idx] = 0
            self.last_check_time[cam_idx] = time.time()
            self.motion_contours[cam_idx] = []

            if cam_idx in self.camera_motion:
                motion_logger.reset_camera_objects(cam_idx)

        motion_logger.log_system_event("Все камеры сброшены в режим ожидания")

    def adjust_sensitivity(self, delta):
        """Изменение чувствительности детекции"""
        old_threshold = self.MOTION_THRESHOLD
        self.MOTION_THRESHOLD = max(5, min(100, self.MOTION_THRESHOLD + delta))
        if old_threshold != self.MOTION_THRESHOLD:
            sensitivity = "увеличена" if delta < 0 else "уменьшена"
            motion_logger.log_system_event(
                f"Чувствительность {sensitivity}: threshold={self.MOTION_THRESHOLD}"
            )

    def cleanup(self):
        """Очистка ресурсов при завершении"""
        # Логирование для активных камер
        for cam_idx in list(self.active_motion_cameras):
            duration = time.time() - self.motion_start_time[cam_idx]
            total_objects = motion_logger.object_counter.get(cam_idx, 0)
            motion_logger.log_motion_stopped(cam_idx, duration, total_objects)

        motion_logger.log_system_event("Завершение работы системы")

        # Освобождение ресурсов
        release_cameras(self.caps)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = SurveillanceSystem()
    system.main_menu()


