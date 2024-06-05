from utils import enumerate_resume, make_printv, write_jsonl, IMPORT_HEADER, count_solved
from executors import executor_factory
from generators import model_factory
from generators import PyGenerator
from executors import PyExecutor
from typing import List
from filelock import FileLock
from multiprocessing import Process, Pool
import logging
import time
from utils import prepare_function_from_seed

def get_seed(i, item, model, num_items, pass_at_k, gen, log_path):
    print(f'[Start] {i+1}')
    exe = executor_factory("python", False)
    cur_pass = 0
    is_solved = False
    cur_func_impl = ""
    num_success = 0
    dataset_type = item["task_id"].split("/")[0]
    token_num = 0
    while cur_pass < pass_at_k:
        cur_func_impl, messages = gen.func_impl(item["prompt"], model, "simple", given_tests=item["given_tests"], dataset_type=dataset_type)
        assert isinstance(cur_func_impl, str)
        if cur_pass > 0:
            # We count the token number only when the first pass is failed to match debugging
            token_num += sum([len(msg.content) for msg in messages])
        cur_func_impl = item["prompt"] + "\n" + cur_func_impl
        cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], cur_func_impl, item["entry_point"])

        is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout = 20)
        if is_solved:
            num_success += 1
            break
        cur_pass += 1

    return cur_func_impl, is_solved, token_num, cur_pass

def async_main(
        dataset: List[dict],
        model_name: str,
        pass_at_k: int,
        n_proc: int,
        log_path: str,
        verbose: bool,
        testfile: str = None,
        logger: logging.Logger = None
    ) -> None:
    gen = PyGenerator()
    exe = PyExecutor()
    # model = model_factory(model_name)
    model = None
    print_v = make_printv(verbose)
    num_items = len(dataset)
    num_success = 0
    if n_proc == 1:
        for i, item in enumerate_resume(dataset, log_path, testfile=testfile):
            logger.info(f"Starting {i+1}th task", extra={"task_id": item["task_id"], "type": "task_started"})
            tests_i = item['given_tests']
            # clean test_i
            tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]
            start_time = time.time()
            step = 0
            while step < 5:
                model = model_factory(model_name, logger=logger, client_type="openai")

                cur_func_impl, is_solved, token_num, cur_pass = get_seed(i, item, model, num_items, pass_at_k, gen, log_path)
                # call the executor to return failed_test
                is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
                if is_passing:
                    break
                step += 1
            end_time = time.time()
            task_time = end_time - start_time
            logger.info(f"Time taken for {i+1}th task: {end_time - start_time}", extra={"task_time": task_time, 
                                                                                        "task_id": item["task_id"],
                                                                                        "type": "task_finished",
                                                                                        "boosting_steps": step,
                                                                                        "model_name": model.name})
            item["solution"] = cur_func_impl
            item["is_solved"] = is_solved
            item['token_num'] = token_num
            item['debug_iter'] = cur_pass
            item['boosting_steps'] = step
            #with FileLock(log_path + ".lock"):
            write_jsonl(log_path, [item], append=True)
            print(f'Completed {i+1}/{num_items}')
        return
    # divide dataset into several groups
    with Pool(n_proc) as pool:
        args = iter([(i, item, model, num_items, pass_at_k, gen, log_path) for i, item in enumerate_resume(dataset, log_path, testfile=testfile)])
        pool.starmap(get_seed, args)

def run_simple_repeat(
        dataset: List[dict],
        model_name: str,
        pass_at_k: int,
        n_proc: int,
        log_path: str,
        verbose: bool,
        testfile: str = None,
        logger: logging.Logger = None
    ) -> None:
    async_main(dataset, model_name, pass_at_k, n_proc, log_path, verbose, testfile, logger)
    print("Accuracy:", count_solved(log_path))
    logger.info("Finished run", extra={"accuracy": count_solved(log_path), "type": "run_finished"})
