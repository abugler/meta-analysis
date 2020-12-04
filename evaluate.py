import json
import os
import glob
from shutil import rmtree
from nussl.evaluation import BSSEvalScale, aggregate_score_files, report_card()
from torch.utils.data import DataLoader


RESULTS_DIR = 'results'

def evaluate(model, dataset, device='cuda'):
    """
    Assumptions:
     - Each model will output a dictionary of estimated sources, as AudioSignals.
       (Therefore, it will be required to write a wrapper for each model)
    """
    dataloader = DataLoader(dataset)
    model_name = type(model).__name__
    dataset_name = type(dataset).__name__

    results_dir = os.path.join(RESULTS_DIR, model_name, dataset_name)
    if os.path.isdir(results_dir):
        rmtree(results_dir)
    os.makedirs(results_dir)

    for idx, batch in enumerate(dataloader):
        _evaluate_one(model, idx, batch, results_dir, device)
    json_files = glob.glob(f"{results_dir}/*.json")
    df = aggregate_score_files(json_files)
    report_card = report_card(
        df, notes="Testing on sine waves", report_each_source=True)
    print(report_card)
    with open(os.path.join(results_dir, 'report_card.txt')) as f:
        f.write(report_card)
    df.to_csv(os.path.join(results_dir, 'aggreggate.csv'))

def _evaluate_one(model, idx, batch, results_dir, device):
    mix = batch['mix'].to(device) # this will be torch tensor.
    sources = batch['sources'] # this will not be
    estimates = model(mix)
    src_list = []
    est_list = []
    for key in sources.keys():
        src_list.append(sources[key])
        est_list.append(estimates[key])
    scores = BSSEvalScale(
        src_list,
        est_list,
        source_labels=list(sources.keys())).evaluate()
    output_path = os.path.join(
        results_dir, f"{idx}.json"
    )
    with open(output_path, 'w') as f:
        json.dump(scores, f)



