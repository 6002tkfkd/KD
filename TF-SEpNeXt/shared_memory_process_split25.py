import os
import time
import numpy as np
import pandas as pd
import librosa
from multiprocessing import shared_memory
from tqdm import tqdm  # tqdm 임포트

# ----------------------------------------
# resource_tracker 공유 메모리 등록 우회 적용
import multiprocessing.resource_tracker

_original_register = multiprocessing.resource_tracker.register

def _fake_register(name, rtype):
    if rtype == 'shared_memory':
        return  # 공유 메모리 등록 무시
    return _original_register(name, rtype)

multiprocessing.resource_tracker.register = _fake_register
# ----------------------------------------

def create_shared_memory_dcase(meta_dir, audio_dir, subset, sampling_rate, unique_labels, shm_prefix="audio_labels_dataset"):
    meta_path = os.path.join(meta_dir, f"{subset}.csv")
    meta_subset = pd.read_csv(meta_path, sep='\t')
    num_items = len(meta_subset)

    # 공유 메모리 이름
    shm_audio_name = f"{shm_prefix}_{subset}_audio"
    shm_scene_name = f"{shm_prefix}_{subset}_scene"
    shm_device_name = f"{shm_prefix}_{subset}_device"
    shm_city_name = f"{shm_prefix}_{subset}_city"

    # 데이터 로드 및 전처리
    wavs = []
    scene_labels = []
    device_labels = []
    city_labels = []

    for _, row in tqdm(meta_subset.iterrows(), total=num_items, desc=f"[{subset}] Loading audio and labels"):
        filename = row["filename"]
        filepath = os.path.join(audio_dir, filename)
        wav, _ = librosa.load(filepath, sr=sampling_rate)
        wavs.append(wav.astype(np.float32))

        scene_str = filename.split('/')[-1].split('-')[0]
        device_str = filename.split('-')[-1].split('.')[0]
        city_str = filename.split('-')[1]

        scene_labels.append(unique_labels['scene'].index(scene_str))
        device_labels.append(unique_labels['device'].index(device_str))
        city_labels.append(unique_labels['city'].index(city_str))

    max_len = max(len(w) for w in wavs)
    audio_arr = np.zeros((num_items, max_len), dtype=np.float32)
    for i, w in enumerate(wavs):
        audio_arr[i, :len(w)] = w

    scene_arr = np.array(scene_labels, dtype=np.int64)
    device_arr = np.array(device_labels, dtype=np.int64)
    city_arr = np.array(city_labels, dtype=np.int64)

    shape_file = f"{subset}_audio_shape.txt"
    with open(shape_file, "w") as f:
        f.write(f"{audio_arr.shape[0]},{audio_arr.shape[1]}\n")

    def safe_unlink(shm_name):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            print(f"[Manager {subset}] Unlinked existing shared memory: {shm_name}")
        except FileNotFoundError:
            pass

    safe_unlink(shm_audio_name)
    safe_unlink(shm_scene_name)
    safe_unlink(shm_device_name)
    safe_unlink(shm_city_name)

    shm_audio = shared_memory.SharedMemory(create=True, size=audio_arr.nbytes, name=shm_audio_name)
    shm_scene = shared_memory.SharedMemory(create=True, size=scene_arr.nbytes, name=shm_scene_name)
    shm_device = shared_memory.SharedMemory(create=True, size=device_arr.nbytes, name=shm_device_name)
    shm_city = shared_memory.SharedMemory(create=True, size=city_arr.nbytes, name=shm_city_name)

    print(f"[Manager {subset}] Copying data to shared memory...")

    shm_audio_array = np.ndarray(audio_arr.shape, dtype=audio_arr.dtype, buffer=shm_audio.buf)
    shm_scene_array = np.ndarray(scene_arr.shape, dtype=scene_arr.dtype, buffer=shm_scene.buf)
    shm_device_array = np.ndarray(device_arr.shape, dtype=device_arr.dtype, buffer=shm_device.buf)
    shm_city_array = np.ndarray(city_arr.shape, dtype=city_arr.dtype, buffer=shm_city.buf)

    for i in tqdm(range(num_items), desc=f"[{subset}] Copying audio data to shared memory"):
        shm_audio_array[i, :len(wavs[i])] = wavs[i]
    shm_scene_array[:] = scene_arr[:]
    shm_device_array[:] = device_arr[:]
    shm_city_array[:] = city_arr[:]

    print(f"[Manager {subset}] Created and initialized shared memories for subset '{subset}'.")


def main():
    META_DIR = "data/meta_dcase_2025"
    AUDIO_DIR = "/data/TAU-urban-acoustic-scenes-2022-mobile-development"
    SAMPLING_RATE = 32000
    UNIQUE_LABELS = {
        'scene': ["airport", "bus", "metro", "metro_station", "park", "public_square",
                  "shopping_mall", "street_pedestrian", "street_traffic", "tram"],
        'device': ["a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6"],
        'city': ["barcelona", "berlin", "london", "paris", "vienna","lisbon", "prague", "lyon", "helsinki", "stockholm", "milan"]
    }

    SUBSETS = ["split25", "valid"]

    for subset in SUBSETS:
        print(f"=== Processing subset: {subset} ===")
        create_shared_memory_dcase(META_DIR, AUDIO_DIR, subset, SAMPLING_RATE, UNIQUE_LABELS)

    print("[Manager] All subsets processed. Holding shared memories alive. Use Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Manager] Cleaning up and exiting...")


if __name__ == "__main__":
    main()
