import os
import argparse
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *

os.environ["MUJOCO_GL"]= "egl"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=["select_fruit"], help="Specific tasks to run, work when eval-track is None")
    parser.add_argument('--eval-track', default=None, help="The evaluation track to run")
    parser.add_argument('--n-episode', default=1, type=int, help="The number of episodes to evaluate for a task")
    parser.add_argument('--policy', default="openvla", help="The policy to evaluate")
    parser.add_argument('--model_ckpt', default="/mnt/data/310_jiarui/VLABench/model_parameter/log/train_log/openvla-7b+vlabench_dataset+b80+lr-0.0005+lora-r16+dropout-0.0--time-20250326-17", help="The base model checkpoint path" )
    parser.add_argument('--lora_ckpt', default="/mnt/data/310_jiarui/VLABench/model_parameter/log/adapter/openvla-7b+vlabench_dataset+b80+lr-0.0005+lora-r16+dropout-0.0--time-20250326-22", help="The lora checkpoint path")
    parser.add_argument('--save-dir', default="logs", help="The directory to save the evaluation results")
    parser.add_argument('--visulization', action="store_true", default=False, help="Whether to visualize the episodes")
    parser.add_argument('--metric', nargs='+', default=["success_rate"], choices=["success_rate", "intention_score", "progress_score"], help="The metrics to evaluate")
    args = parser.parse_args()
    return args

def evaluate(args):
    if args.eval_track is not None:
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", args.eval_track), "r") as f:
            tasks = json.load(f)
    else:
        tasks = args.tasks
    assert isinstance(tasks, list)

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        max_substeps=10,   
        save_dir=args.save_dir,
        visulization=args.visulization,
        metrics=args.metric
    )
    if args.policy.lower() == "openvla":
        policy = OpenVLA(
            model_ckpt=args.model_ckpt,
            lora_ckpt=args.lora_ckpt,
            norm_config_file=os.path.join(os.getenv("VLABENCH_ROOT"), "configs/model/openvla_config.json") # TODO: re-compuate the norm state by your own dataset
        )

    result = evaluator.evaluate(policy)
    with open(os.path.join(args.save_dir, args.policy, "evaluation_result.json"), "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    args = get_args()
    evaluate(args)