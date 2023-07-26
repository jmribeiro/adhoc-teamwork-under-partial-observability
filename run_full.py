from run_adhoc import run
from run_pretrain import load_environment_config, pretrain
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("environment", type=str)
    parser.add_argument("--trials", default=32, type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--resources", default="resources/environments", type=str)
    parser.add_argument("--results", default="resources/results", type=str)
    parser.add_argument("--config", default="config.csv", type=str)
    parser.add_argument('--agents', nargs='+',
                        default=[
                            #"Value_Iteration",
                            #"Perseus",
                            #"Random_Agent",
                            #"ATPO",
                            "BOPA",
                            "PerseusRandom"
                        ])

    opt = parser.parse_args()
    config = load_environment_config(opt.config, opt.environment)

    print("1 - Pre-training...", flush=True)
    pretrain(opt.environment, config, opt.resources, opt.results)

    print("2 - Evaluating...", flush=True)
    opt.agents = [agent.replace("_", " ") for agent in opt.agents]
    run(opt.environment, opt.agents, opt.trials, opt.rerun, config, opt.resources, opt.results)
