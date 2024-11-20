import optuna

# importing eval tools to set up objective
from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow


def parameter_optimizer(
    workflow: BaseNDRagWorkflow, n_trials: int, maximize: bool = True
):
    if maximize:
        direction = "maximize"
    else:
        direction = "minimize"
    study = optuna.create_study(
        study_name=workflow.job_name, direction=direction
    )
    study.optimize(workflow.objective, n_trials=n_trials)

    print(study.best_params)
    return {"best_params": study.best_params, "trials": study.trials}
