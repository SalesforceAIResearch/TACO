import warnings

from .image_base import img_root_map
from .image_caption import ImageCaptionDataset
from .image_yorn import ImageYORNDataset
from .image_mcq import ImageMCQDataset, MMMUDataset, CustomMCQDataset, NaturalBenchDataset
from .image_mt import MMDUDataset
from .image_vqa import (
    ImageVQADataset, MathVision, OCRBench, MathVista, LLaVABench, MMVet, MTVQADataset, CustomVQADataset
)
from .mmbench_video import MMBenchVideo
from .videomme import VideoMME
from .utils import *
from ..smp import *


# Add new supported dataset class here
IMAGE_DATASET = [
    ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, ImageVQADataset, MathVision,
    MMMUDataset, OCRBench, MathVista, LLaVABench, MMVet, MTVQADataset,
    MMLongBench, VCRDataset, MMDUDataset, NaturalBenchDataset
]

VIDEO_DATASET = [
    MMBenchVideo, VideoMME
]

CUSTOM_DATASET = [
    CustomMCQDataset, CustomVQADataset,
]

DATASET_CLASSES = IMAGE_DATASET + VIDEO_DATASET + CUSTOM_DATASET
SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset):
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            return cls.TYPE


def build_dataset(dataset_name, **kwargs):
    for cls in (IMAGE_DATASET + VIDEO_DATASET):
        if dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')

    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    if 'A' in data and 'B' in data:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
        return CustomMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


__all__ = [
    'build_dataset', 'img_root_map', 'build_judge', 'extract_answer_from_item', 'prefetch_answer', 'DEBUG_MESSAGE'
] + [cls.__name__ for cls in DATASET_CLASSES]
