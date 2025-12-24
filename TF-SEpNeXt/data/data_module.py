import librosa
import numpy as np
import torch
import lightning as L
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util import unique_labels
import os
from multiprocessing import shared_memory, resource_tracker
import json

def attach_shm(name: str) -> shared_memory.SharedMemory:
    """
    SharedMemory(name) 로 붙은 뒤 resource_tracker 자동 unlink를 비활성화한다.
    """
    shm = shared_memory.SharedMemory(name=name, create=False)
    # 🔴 자동 unlink 방지 (중요)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm

def load_meta(subset: str):
    """
    producer가 저장한 {subset}.meta.json 읽기.
    없으면 None 반환.
    """
    meta_path = f"{subset}.meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_audio_shape_from_txt(subset: str):
    with open(f"{subset}_audio_shape.txt", "r") as f:
        line = f.readline().strip()
        nums = [int(x) for x in line.split(",")]
    if len(nums) != 2:
        raise RuntimeError(f"Invalid shape txt for subset {subset}: {nums}")
    return tuple(nums)

class AudioDataset(Dataset):
    """
    Dataset containing pairs of audio waveform and filename.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        sampling_rate (int): Sampling rate of waveforms.
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')

    def __len__(self):
        return len(self.meta_subset)

    def __getitem__(self, i):
        # Get the ith row from the meta csv file
        row_i = self.meta_subset.iloc[i]
        # Get the filename of audio
        filename = row_i["filename"]
        # Load audio waveform with a resample rate
        wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav = torch.from_numpy(wav)
        return wav, filename

class AudioDatasetCached(Dataset):
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')

        self.data = []
        for idx, row in self.meta_subset.iterrows():
            filename = row["filename"]
            filepath = os.path.join(self.audio_dir, filename)
            wav, _ = librosa.load(filepath, sr=self.sr)
            wav_tensor = torch.from_numpy(wav)
            self.data.append((wav_tensor, filename))
        print(f"[INFO] Cached {len(self.data)} items into memory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class AudioDatasetShared(Dataset):
    """
    waveform만 shared memory에서 읽어오는 소비자 전용 Dataset
    (소비자는 절대 생성 X, 없으면 에러)
    """

    def __init__(self, meta_dir: str, audio_dir: str, subset: str,
                 sampling_rate: int = 16000, shm_prefix: str = "audio_dataset"):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.shm_prefix = shm_prefix

        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')
        self.num_items = len(self.meta_subset)
        self.filenames = list(self.meta_subset["filename"])

        shm_audio_name = f"{self.shm_prefix}_{self.subset}_audio"

        # 메타 우선(JSON), 없으면 txt 폴백
        meta = load_meta(self.subset)
        if meta is not None and "audio" in meta:
            audio_shape = tuple(meta["audio"]["shape"])
            audio_dtype = np.dtype(meta["audio"]["dtype"])
        else:
            audio_shape = load_audio_shape_from_txt(self.subset)
            audio_dtype = np.float32

        # 🔴 attach only (create=False) + unregister
        self.shm_audio = attach_shm(shm_audio_name)

        self.audio = np.ndarray(audio_shape, dtype=audio_dtype, buffer=self.shm_audio.buf)
        print(f"[INFO] Attached audio shared memory for subset '{self.subset}' with shape {self.audio.shape}")

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        wav_tensor = torch.tensor(self.audio[i], dtype=torch.float32)
        filename = self.filenames[i]
        return wav_tensor, filename


class AudioLabelsDataset(AudioDataset):
    """
    Dataset containing tuples of audio waveform, scene label, device label and city label.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        sampling_rate (int): Sampling rate of waveforms.
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        super().__init__(meta_dir, audio_dir, subset, sampling_rate)

    def __getitem__(self, i):
        # Get the filename
        wav, filename = super().__getitem__(i)
        scene_label = filename.split('/')[-1].split('-')[0]
        device_label = filename.split('-')[-1].split('.')[0]
        city_label = filename.split('-')[1]
        # Encode the scene labels from string to integers
        scene_label = unique_labels['scene'].index(scene_label)
        scene_label = torch.from_numpy(np.array(scene_label, dtype=np.int64))
        # Encode the device labels from string to integers
        device_label = unique_labels['device'].index(device_label)
        device_label = torch.from_numpy(np.array(device_label, dtype=np.int64))
        # Encode the city labels from string to integers
        city_label = unique_labels['city'].index(city_label)
        city_label = torch.from_numpy(np.array(city_label, dtype=np.int64))
        return wav, scene_label, device_label, city_label

class AudioLabelsDatasetCached(Dataset):
    """
    Cached dataset: audio waveform, scene label, device label and city label.
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate

        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')

        self.data = []
        for _, row in self.meta_subset.iterrows():
            filename = row["filename"]
            filepath = os.path.join(self.audio_dir, filename)
            wav, _ = librosa.load(filepath, sr=self.sr)
            wav_tensor = torch.from_numpy(wav)

            scene_label_str = filename.split('/')[-1].split('-')[0]
            device_label_str = filename.split('-')[-1].split('.')[0]
            city_label_str = filename.split('-')[1]

            scene_label = torch.tensor(unique_labels['scene'].index(scene_label_str), dtype=torch.long)
            device_label = torch.tensor(unique_labels['device'].index(device_label_str), dtype=torch.long)
            city_label = torch.tensor(unique_labels['city'].index(city_label_str), dtype=torch.long)

            self.data.append((wav_tensor, scene_label, device_label, city_label))
        print(f"[INFO] Cached {len(self.data)} items into memory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class AudioLabelsDatasetShared(Dataset):
    """
    waveform + scene/device/city 레이블을 shared memory에서 읽는 소비자용 Dataset.
    (없으면 에러; 생성하지 않음)
    """

    def __init__(self, meta_dir: str, audio_dir: str, subset: str,
                 sampling_rate: int = 16000, unique_labels: dict = None,
                 shm_prefix: str = "audio_labels_dataset"):
        assert unique_labels is not None, "unique_labels must be provided"

        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.unique_labels = unique_labels
        self.shm_prefix = shm_prefix

        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')
        self.num_items = len(self.meta_subset)

        # shm names
        shm_audio_name  = f"{self.shm_prefix}_{self.subset}_audio"
        shm_scene_name  = f"{self.shm_prefix}_{self.subset}_scene"
        shm_device_name = f"{self.shm_prefix}_{self.subset}_device"
        shm_city_name   = f"{self.shm_prefix}_{self.subset}_city"

        # 메타 JSON or txt
        meta = load_meta(self.subset)
        if meta is not None and "audio" in meta:
            audio_shape = tuple(meta["audio"]["shape"])
            audio_dtype = np.dtype(meta["audio"]["dtype"])
            n_items = meta.get("n_items", self.num_items)
            if n_items != self.num_items:
                print(f"[WARN] meta n_items({n_items}) != csv rows({self.num_items})")
        else:
            audio_shape = load_audio_shape_from_txt(self.subset)
            audio_dtype = np.float32

        # 🔴 attach only + unregister(자동 unlink 방지)
        self.shm_audio  = attach_shm(shm_audio_name)
        self.shm_scene  = attach_shm(shm_scene_name)
        self.shm_device = attach_shm(shm_device_name)
        self.shm_city   = attach_shm(shm_city_name)

        # numpy view
        self.audio         = np.ndarray(audio_shape,                dtype=audio_dtype, buffer=self.shm_audio.buf)
        self.scene_labels  = np.ndarray((self.num_items,),         dtype=np.int64,    buffer=self.shm_scene.buf)
        self.device_labels = np.ndarray((self.num_items,),         dtype=np.int64,    buffer=self.shm_device.buf)
        self.city_labels   = np.ndarray((self.num_items,),         dtype=np.int64,    buffer=self.shm_city.buf)

        print(f"[INFO] Attached shared memory for '{self.subset}' (audio {self.audio.shape}, labels {self.num_items})")

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        wav_tensor    = torch.tensor(self.audio[i], dtype=torch.float32)
        scene_label   = torch.tensor(self.scene_labels[i], dtype=torch.long)
        device_label  = torch.tensor(self.device_labels[i], dtype=torch.long)
        city_label    = torch.tensor(self.city_labels[i], dtype=torch.long)
        return wav_tensor, scene_label, device_label, city_label

class AudioLabelsDatasetWithLogits(AudioLabelsDataset):
    """
    AudioLabelsDataset with additional logits of teacher ensemble for knowledge distillation.

    Args:
        logits_files (list): List of directories of teacher logits. e.g. ["path/to/logit/predictions.pt", ...]
    """
    def __init__(self, logits_files: list, **kwargs):
        super().__init__(**kwargs)
        logits_all = []
        for file in logits_files:
            # Load teacher logit and append to a list
            logit = torch.load(file).float()
            logits_all.append(logit)
        # Average the logits from multiple teachers
        logit_all = sum(logits_all)
        self.teacher_logit = logit_all / len(logits_files)

    def __getitem__(self, i):
        wav, scene_label, device_label, city_label = super().__getitem__(i)
        return wav, scene_label, device_label, city_label, self.teacher_logit[i]

class AudioLabelsDatasetWithLogitsCached(Dataset):
    """
    Cached dataset with teacher logits for knowledge distillation.
    Loads all audio, labels, and logits into memory during initialization.

    Args:
        logits_files (list): List of paths to teacher logits .pt files.
        meta_dir (str): Directory of meta files.
        audio_dir (str): Directory of audio files.
        subset (str): Dataset split (e.g., train, valid).
        sampling_rate (int): Audio sampling rate.
    """
    def __init__(self, logits_files: list, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate

        # Load meta
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')

        # Load and average logits
        logits_all = [torch.load(f).float() for f in logits_files]
        self.teacher_logit = sum(logits_all) / len(logits_all)

        # Preload data
        self.data = []
        for i, row in self.meta_subset.iterrows():
            filename = row["filename"]
            filepath = os.path.join(self.audio_dir, filename)
            wav, _ = librosa.load(filepath, sr=self.sr)
            wav_tensor = torch.from_numpy(wav)

            # Label parsing
            scene_str = filename.split('/')[-1].split('-')[0]
            device_str = filename.split('-')[-1].split('.')[0]
            city_str = filename.split('-')[1]

            scene_label = torch.tensor(unique_labels['scene'].index(scene_str), dtype=torch.long)
            device_label = torch.tensor(unique_labels['device'].index(device_str), dtype=torch.long)
            city_label = torch.tensor(unique_labels['city'].index(city_str), dtype=torch.long)

            self.data.append((wav_tensor, scene_label, device_label, city_label, self.teacher_logit[i]))
        print(f"[INFO] Cached {len(self.data)} items into memory with logits.")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class AudioLabelsDatasetWithLogitsShared(Dataset):
    """
    waveform + scene/device/city + logits 를 shared memory에서 읽는 소비자용 Dataset.
    (없으면 에러; 생성하지 않음)
    """
    def __init__(self, logits_files: list, meta_dir: str, audio_dir: str,
                 subset: str, sampling_rate: int = 16000, unique_labels: dict = None,
                 shm_prefix: str = "audio_labels_dataset"):
        assert unique_labels is not None, "unique_labels must be provided"
        assert isinstance(logits_files, (list, tuple)) and len(logits_files) > 0, "logits_files must be a non-empty list"

        self.logits_files = logits_files
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.unique_labels = unique_labels
        self.shm_prefix = shm_prefix

        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')
        self.num_items = len(self.meta_subset)

        shm_audio_name  = f"{self.shm_prefix}_{self.subset}_audio"
        shm_scene_name  = f"{self.shm_prefix}_{self.subset}_scene"
        shm_device_name = f"{self.shm_prefix}_{self.subset}_device"
        shm_city_name   = f"{self.shm_prefix}_{self.subset}_city"
        shm_logits_name = f"{self.shm_prefix}_{self.subset}_logits"

        # 메타 JSON or txt
        meta = load_meta(self.subset)
        if meta is not None and "audio" in meta:
            audio_shape = tuple(meta["audio"]["shape"])
            audio_dtype = np.dtype(meta["audio"]["dtype"])
            n_items = meta.get("n_items", self.num_items)
            if n_items != self.num_items:
                print(f"[WARN] meta n_items({n_items}) != csv rows({self.num_items})")
        else:
            audio_shape = load_audio_shape_from_txt(self.subset)
            audio_dtype = np.float32

        # logits 클래스 수 계산(파일에서 확인)
        num_classes = self._logits_num_classes_from_files(self.logits_files)

        # 🔴 attach only + unregister
        self.shm_audio  = attach_shm(shm_audio_name)
        self.shm_scene  = attach_shm(shm_scene_name)
        self.shm_device = attach_shm(shm_device_name)
        self.shm_city   = attach_shm(shm_city_name)
        self.shm_logits = attach_shm(shm_logits_name)

        # numpy views
        self.audio         = np.ndarray(audio_shape,                      dtype=audio_dtype, buffer=self.shm_audio.buf)
        self.scene_labels  = np.ndarray((self.num_items,),               dtype=np.int64,    buffer=self.shm_scene.buf)
        self.device_labels = np.ndarray((self.num_items,),               dtype=np.int64,    buffer=self.shm_device.buf)
        self.city_labels   = np.ndarray((self.num_items,),               dtype=np.int64,    buffer=self.shm_city.buf)
        self.logits        = np.ndarray((self.num_items, num_classes),   dtype=np.float32,  buffer=self.shm_logits.buf)

        print(f"[INFO] Attached shared memory for '{self.subset}' (audio {self.audio.shape}, logits {self.logits.shape})")

    @staticmethod
    def _logits_num_classes_from_files(files):
        # 첫 파일 하나 열어 shape[1] 사용
        sample = torch.load(files[0])
        if sample.ndim != 2:
            raise ValueError(f"Logits must be 2-D (N, C). Got shape {tuple(sample.shape)}")
        return int(sample.shape[1])

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        wav_tensor    = torch.tensor(self.audio[i], dtype=torch.float32)
        scene_label   = torch.tensor(self.scene_labels[i], dtype=torch.long)
        device_label  = torch.tensor(self.device_labels[i], dtype=torch.long)
        city_label    = torch.tensor(self.city_labels[i], dtype=torch.long)
        logits_tensor = torch.tensor(self.logits[i], dtype=torch.float32)
        return wav_tensor, scene_label, device_label, city_label, logits_tensor


class DCASEDataModule(L.LightningDataModule):
    """
    DCASE DataModule wrapping train, validation, test and predict DataLoaders.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use for DataLoaders. Will save time for loading data to GPU but increase CPU usage.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Will save time for data loading.
        logits_files (list): List of directories of teacher logits, e.g. ["path/to/logit/predictions.pt", ...]. If not ``None``, knowledge distillation will be applied.
        train_subset (str): Name of train meta file. e.g. train, split5, split10...
        test_subset (str): Name of test meta file.
        predict_subset (str): Name of predict meta file.
    """
    def __init__(self, meta_dir: str, audio_dir: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool=False,
                 logits_files=None, train_subset="train", test_subset="test", predict_subset="test", **kwargs):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.logits_files = logits_files
        self.kwargs = kwargs

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Add teacher logits to the dataset if using knowledge distillation
            if self.logits_files is not None:
                self.train_set = AudioLabelsDatasetWithLogits(logits_files=self.logits_files, meta_dir=self.meta_dir, audio_dir=self.audio_dir, subset=self.train_subset, **self.kwargs)
            else:
                self.train_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)
        if stage == "validate":
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)
        if stage == "test":
            self.test_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs)
        if stage == "predict":
            self.predict_set = AudioDataset(self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

class DCASEDataModuleCached(L.LightningDataModule):
    """
    DCASE DataModule wrapping train, validation, test and predict DataLoaders.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use for DataLoaders. Will save time for loading data to GPU but increase CPU usage.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Will save time for data loading.
        logits_files (list): List of directories of teacher logits, e.g. ["path/to/logit/predictions.pt", ...]. If not ``None``, knowledge distillation will be applied.
        train_subset (str): Name of train meta file. e.g. train, split5, split10...
        test_subset (str): Name of test meta file.
        predict_subset (str): Name of predict meta file.
    """
    def __init__(self, meta_dir: str, audio_dir: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool=False,
                 logits_files=None, train_subset="train", test_subset="test", predict_subset="test", **kwargs):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.logits_files = logits_files
        self.kwargs = kwargs

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Add teacher logits to the dataset if using knowledge distillation
            if self.logits_files is not None:
                self.train_set = AudioLabelsDatasetWithLogitsCached(logits_files=self.logits_files, meta_dir=self.meta_dir, audio_dir=self.audio_dir, subset=self.train_subset, **self.kwargs)
            else:
                self.train_set = AudioLabelsDatasetCached(self.meta_dir, self.audio_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDatasetCached(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)
        if stage == "validate":
            self.valid_set = AudioLabelsDatasetCached(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)
        if stage == "test":
            self.test_set = AudioLabelsDatasetCached(self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs)
        if stage == "predict":
            self.predict_set = AudioDatasetCached(self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)


class DCASEDataModuleCachedShared(L.LightningDataModule):
    def __init__(self, meta_dir: str, audio_dir: str, batch_size: int = 16, num_workers: int = 0,
                 pin_memory: bool = False, logits_files=None, train_subset="train", test_subset="test",
                 predict_subset="test", unique_labels: dict = None, **kwargs):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.logits_files = logits_files
        self.unique_labels = unique_labels if unique_labels is not None else {'scene': [
                                                                                "airport", "bus", "metro", "metro_station", "park", "public_square",
                                                                                "shopping_mall", "street_pedestrian", "street_traffic", "tram"],
                                                                            'device': ["a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6"],
                                                                            'city': ["barcelona", "berlin", "london", "paris", "vienna","lisbon", "prague", "lyon", "helsinki", "stockholm", "milan"]}
        self.kwargs = kwargs

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            if self.logits_files is not None:
                self.train_set = AudioLabelsDatasetWithLogitsShared(
                    logits_files=self.logits_files,
                    meta_dir=self.meta_dir,
                    audio_dir=self.audio_dir,
                    subset=self.train_subset,
                    unique_labels=self.unique_labels,
                    **self.kwargs,
                )
            else:
                self.train_set = AudioLabelsDatasetShared(
                    meta_dir=self.meta_dir,
                    audio_dir=self.audio_dir,
                    subset=self.train_subset,
                    unique_labels=self.unique_labels,
                    **self.kwargs,
                )
            self.valid_set = AudioLabelsDatasetShared(
                meta_dir=self.meta_dir,
                audio_dir=self.audio_dir,
                subset="valid",
                unique_labels=self.unique_labels,
                **self.kwargs,
            )
        if stage == "validate" or stage is None:
            self.valid_set = AudioLabelsDatasetShared(
                meta_dir=self.meta_dir,
                audio_dir=self.audio_dir,
                subset="valid",
                unique_labels=self.unique_labels,
                **self.kwargs,
            )
        if stage == "test" or stage is None:
            self.test_set = AudioLabelsDatasetShared(
                meta_dir=self.meta_dir,
                audio_dir=self.audio_dir,
                subset=self.test_subset,
                unique_labels=self.unique_labels,
                **self.kwargs,
            )
        if stage == "predict" or stage is None:
            self.predict_set = AudioDatasetShared(
                meta_dir=self.meta_dir,
                audio_dir=self.audio_dir,
                subset=self.predict_subset,
                **self.kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
