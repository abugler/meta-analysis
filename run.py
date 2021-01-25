import argbind
from typing import List
import dataset as D
import load_models
from evaluate import evaluate


@argbind.bind()
def run(
    # Dataset Parameters
    segmented: int = 1,
    segment_length: int = 352_800,
    segment_hop: int = 176_400,
    top_db: float = 8.0,
    load_segmentation: int = 1,
    save_segmentation: int = 0,
    toy: int = None,
    subset: str = 'test',
    # Model Parameters
    model_name: str = None,
    device: str = 'cuda',
    # Evaluation Parameters
    num_workers: int = 0
):
    """
    Interface for evaluating SOTA source separation networks.
    """
    segmented = bool(segmented)
    load_segmentation = bool(load_segmentation)
    save_segmentation = bool(save_segmentation)

    model_func = getattr(load_models, model_name)
    if model_func is load_models.wav_u_net:
        model = model_func(segment_length)
    else:
        model = model_func()
    model.to(device)

    if segmented:
        dataset = D.MUSDB18Segmented(
            folder=D.root,
            is_wav=False,
            load_seg=load_segmentation,
            save_seg=save_segmentation,
            segment_length=segment_length,
            segment_hop=segment_hop,
            top_db=top_db,
            toy=toy,
            subset=subset
        )
    else:
        raise NotImplementedError("""
            Evaluating on full MUSDB tracks not implemented yet.
        """)

    evaluate(model, dataset, num_workers=num_workers, device=device)

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        run()
