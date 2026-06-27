import lightning.pytorch as pl
import torch

import wandb
from tnp.utils.experiment_utils import initialize_evaluation, val_epoch
from tnp.utils.lightning_utils import LitWrapper


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.test

    model.eval()

    # Store number of parameters.
    num_params = sum(p.numel() for p in model.parameters())

    if experiment.misc.lightning_eval:
        lit_model = LitWrapper(model)
        trainer = pl.Trainer(devices=1)
        trainer.test(model=lit_model, dataloaders=gen_test)
        test_result = {
            k: [result[k] for result in lit_model.test_outputs]
            for k in lit_model.test_outputs[0].keys()
        }
        loglik = torch.stack(test_result["loglik"])
        test_result["mean_loglik"] = loglik.mean()
        test_result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)

        if "gt_loglik" in test_result:
            gt_loglik = torch.stack(test_result["gt_loglik"])
            test_result["mean_gt_loglik"] = gt_loglik.mean()
            test_result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)

    else:
        test_result, _ = val_epoch(model=model, generator=gen_test)

    if experiment.misc.logging:
        wandb.run.summary["num_params"] = num_params
        wandb.run.summary[f"test/{eval_name}/loglik"] = test_result["mean_loglik"]
        wandb.run.summary[f"test/{eval_name}/std_loglik"] = test_result["std_loglik"]
        if "mean_gt_loglik" in test_result:
            wandb.run.summary[f"test/{eval_name}/gt_loglik"] = test_result[
                "mean_gt_loglik"
            ]
            wandb.run.summary[f"test/{eval_name}/std_gt_loglik"] = test_result[
                "std_gt_loglik"
            ]


if __name__ == "__main__":
    main()
