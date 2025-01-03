import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--result-file', type=str, required=True)
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    parser.add_argument('--max-tokens', type=int, default=2048)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')

    args = parse_args()
    assert len(args.data), '--data should be a list of data files'

    if args.retry is not None:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

    for _, dataset_name in enumerate(args.data):
        dataset_kwargs = {}

        # If distributed, first build the dataset on the main process for doing preparation works
        if world_size > 1:
            dataset = build_dataset(dataset_name, **dataset_kwargs) if rank == 0 else None
            dist.barrier()
            dataset_list = [dataset]
            dist.broadcast_object_list(dataset_list, src=0)
            dataset = dataset_list[0]
        else:
            dataset = build_dataset(dataset_name, **dataset_kwargs)
        if dataset is None:
            logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
            continue

        result_file = args.result_file 

        # Set the judge kwargs first before evaluation or dumping
        judge_kwargs = {
            'nproc': args.nproc,
            'verbose': args.verbose,
            'max_tokens': args.max_tokens,
        }
        if args.retry is not None:
            judge_kwargs['retry'] = args.retry
        if args.judge is not None:
            judge_kwargs['model'] = args.judge
        else:
            if dataset.TYPE in ['MCQ', 'Y/N']:
                judge_kwargs['model'] = 'chatgpt-0125'
            elif listinstr(['MMVet', 'MathVista', 'LLaVABench', 'MMBench-Video', 'MathVision'], dataset_name):
                judge_kwargs['model'] = 'gpt-4-turbo'
            elif listinstr(['MMLongBench'], dataset_name):
                judge_kwargs['model'] = 'gpt-4o'
        if 'OPENAI_API_KEY_JUDGE' in os.environ and len(os.environ['OPENAI_API_KEY_JUDGE']):
            judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
        if 'OPENAI_API_BASE_JUDGE' in os.environ and len(os.environ['OPENAI_API_BASE_JUDGE']):
            judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

        if rank == 0:
            if dataset_name in ['MMMU_TEST']:
                result_json = MMMU_result_transfer(result_file)
                logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                            f'json file saved in {result_json}')  # noqa: E501
                continue
            elif 'MMT-Bench_ALL' in dataset_name:
                submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                            f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                            f'submission file saved in {submission_file}')  # noqa: E501
                continue
            elif 'MLLMGuard_DS' in dataset_name:
                logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')  # noqa: E501
                continue
            elif 'AesBench_TEST' == dataset_name:
                logger.info(f'The results are saved in {result_file}. '
                            f'Please send it to the AesBench Team via huangyipo@hotmail.com.')  # noqa: E501
                continue

        if dataset_name in [
            'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
            'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
        ]:
            if not MMBenchOfficialServer(dataset_name):
                logger.error(
                    f'Can not evaluate {dataset_name} on non-official servers, '
                    'will skip the evaluation. '
                )
                continue

        eval_proxy = os.environ.get('EVAL_PROXY', None)
        old_proxy = os.environ.get('HTTP_PROXY', '')

        if rank == 0:
            if eval_proxy is not None:
                proxy_set(eval_proxy)

            eval_results = dataset.evaluate(result_file, **judge_kwargs)
            if eval_results is not None:
                assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                logger.info(f'The evaluation on dataset {dataset_name} has finished! ')
                logger.info('Evaluation Results:')
            if isinstance(eval_results, dict):
                logger.info('\n' + json.dumps(eval_results, indent=4))
            elif isinstance(eval_results, pd.DataFrame):
                if len(eval_results) < len(eval_results.columns):
                    eval_results = eval_results.T
                logger.info('\n' + tabulate(eval_results))

            if eval_proxy is not None:
                proxy_set(old_proxy)


if __name__ == '__main__':
    load_env()
    main()
