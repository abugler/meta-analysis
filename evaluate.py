import json
import time
import os
import glob
from tqdm import tqdm
from shutil import rmtree
# from concurrent.futures import ThreadPoolExecutor
from nussl.evaluation import BSSEvalScale, BSSEvalV4, aggregate_score_files, report_card
from torch.utils.data import DataLoader

from numpy.linalg import LinAlgError

RESULTS_DIR = 'results'

def evaluate(model, dataset, num_workers=0, device='cuda'):
    """
    Assumptions:
     - Each model will output a dictionary of estimated sources, as AudioSignals.
       (Therefore, it will be required to write a wrapper for each model)
     - The model should already be on the desired device.
    """
    model_name = model.name
    dataset_name = type(dataset).__name__

    results_dir = os.path.join(RESULTS_DIR, model_name, dataset_name)
    if os.path.isdir(results_dir):
        destroy = None
        while destroy not in ["yes", "no"]:
            destroy = input("Previous results found. Delete them? (Type 'yes' or 'no')")
        if destroy == "no":
            print("Ending run...")
            return 
        rmtree(results_dir)
    os.makedirs(results_dir)
    # with ThreadPoolExecutor(max_workers=num_workers) as pool:
    #     futures = []
    for idx, batch in tqdm(enumerate(dataset),
                        desc="Evaluating",
                        total=len(dataset)):
        _evaluate_one(model, idx, batch, results_dir, device)
        # TODO: Concurrency was giving me a hard time...
        #       Fix it later
        # future = pool.submit(
        #     _evaluate_one, model, idx, batch, results_dir, device
        # )
        #     futures.append(future)
        # print("Waiting for threads to finish...")
        # while futures:
        #     while futures and futures[0].done():
        #         futures.pop(0)
        #     time.sleep(1)
        #     print(f"{len(futures)} jobs remaining...")
    json_files = glob.glob(f"{results_dir}/*.json")
    df = aggregate_score_files(json_files)
    report = report_card(
        df, notes=f"Testing model {type(model_name)}", report_each_source=True)
    print(report)
    with open(os.path.join(results_dir, 'report_card.txt'), 'a') as f:
        f.write(report)
    df.to_csv(os.path.join(results_dir, 'aggreggate.csv'))

def _evaluate_one(model, idx, batch, results_dir, device):
    mix = batch['mix'].to(device) # this will be torch tensor.
    sources = batch['sources'] # this will not be
    estimates = model(mix)
    source_names = list(sources.keys())
    src_list = []
    est_list = []
    # TODO: When one of the sources is empty, identify it,
    #       and label it as a broken example.
    for key in source_names:
        src_list.append(sources[key])
        est_list.append(estimates[key])
    try:
        scores = BSSEvalV4(
            src_list, est_list,
            source_labels=source_names,
            compute_permutation=False).evaluate()
        scale_scores = BSSEvalScale(
            src_list,
            est_list,
            source_labels=source_names,
            compute_permutation=False).evaluate()
    except (LinAlgError, ValueError):
        # a source must be empty...
        # code may also reach this point if the estimated and ground truth are not the same
        # length. In this case, crop the input signal to be the same length as the output.
        return
    for source in scores.keys():
        if source in ['combination', 'permutation']:
            continue
        scores[source].update(scale_scores[source])
    output_path = os.path.join(
        results_dir, f"{idx}.json"
    )
    with open(output_path, 'w') as f:
        json.dump(scores, f)

