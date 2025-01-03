import os 
import torch

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

CACHE_DIR = os.environ["HF_HUB_CACHE"] 
MANTIS_IMAGE_PATH = os.environ["MANTIS_DATA_PATH"]
MANTIS_CKPT_PATH = os.environ["MANTIS_CKPT_PATH"]
LLAVA_OV_CKPT_PATH = os.environ["LLAVA_OV_CKPT_PATH"]

INPUT_IMAGE_PATH = "/export/share/zixianma/data/mma_execute/inputs"

RESULT_PATH = "/export/share/zixianma/data/mma_execute/results"

TRAIN_DATA_PATH = "/export/agentstudio-family/zixian/data"

CACHE_ROOT_PATH = "/gscratch/krishna/zixianma"

MANTIS_DATA_PATH = "/export/share/jieyu/mantis_data"

LATEST_GPT_MODEL_ID = "gpt-4o-2024-08-06"

EVAL_IMAGE_BASE_PATH = "/export/agentstudio-family/zixian/taco/eval_data/images"
            
DATASET_BASE_PATH = "/export/agentstudio-family/zixian/taco/eval_data"
DATASET_TO_PATH = {
    "RealWorldQA": "/export/agentstudio-family/zixian/realworldqa/realworldqa_with_options.jsonl",
    "MMVet": f"{DATASET_BASE_PATH}/MMVet.tsv",
    "BLINK": f"{DATASET_BASE_PATH}/BLINK.tsv",
    "MMVP": f"{DATASET_BASE_PATH}/MMVP.tsv",
    "MMStar": f"{DATASET_BASE_PATH}/MMStar.tsv",
    "MathVista_MINI": f"{DATASET_BASE_PATH}/MathVista_MINI.tsv",
    "A-OKVQA": f"{DATASET_BASE_PATH}/A-OKVQA.tsv",
    "MMMU_DEV_VAL": f"{DATASET_BASE_PATH}/MMMU_DEV_VAL.tsv",
    "TextVQA_VAL": f"{DATASET_BASE_PATH}/TextVQA_VAL.tsv",
    "HallusionBench": f"{DATASET_BASE_PATH}/HallusionBench.tsv",
    "NaturalBenchDataset": f"{DATASET_BASE_PATH}/NaturalBenchDataset.tsv",
    "GQA_TestDev_Balanced": f"{DATASET_BASE_PATH}/GQA_TestDev_Balanced.tsv",
    "POPE": f"{DATASET_BASE_PATH}/POPE.tsv",
    "TaskMeAnything_v1_imageqa_random": f"{DATASET_BASE_PATH}/TaskMeAnything_v1_imageqa_random.tsv",
    "MME": "/export/agentstudio-family/zixian/VLMEvalKit/GPT4o/GPT4o_MME_auxmatch.xlsx",
    "CV-Bench": "/export/agentstudio-family/zixian/VLMEvalKit/data/CV-Bench-Sampled.tsv",
    "AI2D_TEST": "/export/agentstudio-family/zixian/VLMEvalKit/data/AI2D_TEST-Sampled.tsv",
    "SEEDBench_IMG": "/export/agentstudio-family/zixian/VLMEvalKit/data/SEEDBench_IMG-Sampled.tsv",
}

SI_DATASETS = ["ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "clevr_math", "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa", "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa", "infographic_vqa", "intergps", "localized_narratives", "mapqa", "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa", "okvqa", "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq", "scienceqa", "screen2words", "spot_the_diff", "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr", "websight"]

MI_DATASETS = ["mantis-birds-to-words", "mantis-coinstruct", "mantis-contrastive_caption", "mantis-dreamsim", "mantis-iconqa", "mantis-imagecode", "mantis-lrv_multi", "mantis-multi_vqa", "mantis-nlvr2", "mantis-spot-the-diff", "mantis-nextqa", "mantis-star"]
MI_DATASETS += [f"mantis-coinstruct-{i}" for i in range(1, 17)] + [f"mantis-llava_665k_multi-{i}" for i in range(1, 17)]
ALL_DATASETS = SI_DATASETS + MI_DATASETS
BAD_DATASETS = ["mantis-dreamsim", "mantis-spot-the-diff", "ai2d", "scienceqa", "mantis-nextqa", "mantis-multi_vqa", "dvqa", "mantis-imagecode", "vqarad"]
GOOD_DATASETS = [ds for ds in ALL_DATASETS if ds not in BAD_DATASETS and not ds.startswith("mantis-coinstruct")]