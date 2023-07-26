import numpy as np
from scipy.stats import entropy
from yaaf.visualization import LinePlot
from run_beliefs_analysis import load_beliefs
from run_pretrain import load_environment_config


def plot_beliefs(agent, environment, results):

    horizon = results[0][1].shape[0]

    plot = LinePlot(f"{agent} on {environment}", "Timestep", "p(m)", horizon, ymax=1.2)

    for m, beliefs in results:

        for m2, model_belief in enumerate(beliefs.T):
            if m2 == m:
                model_tag = "Correct Model"
            else:
                if m2 > m:
                    model_tag = f"Wrong Model #{m2}"
                else:
                    model_tag = f"Wrong Model #{m2+1}"

            plot.add_run(model_tag, model_belief, "green" if model_tag == "Correct Model" else "red")

    plot.show()


def plot_entropy(environment, results, colors):

    horizon = results[list(results.keys())[0]][0][1].shape[0]

    plot = LinePlot(environment, "Timestep", "Entropy p(m)", horizon, ymax=1.2)

    for agent in results:

        for m, beliefs in results[agent]:

            run = np.zeros(horizon)
            for t, timestep_belief in enumerate(beliefs):
                run[t] = entropy(timestep_belief, base=timestep_belief.size)
            plot.add_run(agent, run, color=colors[agent])

    plot.show()


def plot_beliefs_environment(environment, agents, config, results, show_beliefs, show_entropy):

    print(f"{environment}", flush=True)

    env_config = load_environment_config(config, environment)

    beliefs_per_agent = {}
    for agent in agents:
        print(f"\t{agent}:", flush=True, end=" ")
        beliefs = load_beliefs(agent, environment, env_config["noise"], results)
        if len(beliefs) > 0:
            if show_beliefs: plot_beliefs(agent, environment, beliefs)
            beliefs_per_agent[agent] = beliefs
            print(f"Found {len(beliefs)} trials!", flush=True)
        else:
            print(f"No results!", flush=True)

    if len(beliefs_per_agent) > 0 and show_entropy:
        plot_entropy(environment, beliefs_per_agent, agents)
    else:
        print(f"(No results on {environment})", flush=True)


def plot_mean_beliefs_environments(environments, agents, config, results, show_entropy):

    max_horizon = 85
    plot = LinePlot("Entropy", "Timestep", "Entropy", max_horizon+1, ymax=1.2)

    for environment in environments:

        print(f"{environment}", flush=True)

        env_config = load_environment_config(config, environment)

        for agent in agents:
            print(f"\t{agent}:", flush=True, end=" ")
            beliefs = load_beliefs(agent, environment, env_config["noise"], results)
            if len(beliefs) > 0:
                for m, belief in beliefs:
                    run = []
                    for t, timestep_belief in enumerate(belief):
                        run.append(entropy(timestep_belief, base=env_config["models"]))
                    plot.add_run(agent, np.array(run), color=agents[agent])
                print(f"Found {len(beliefs)} trials!", flush=True)
            else:
                print(f"No results!", flush=True)
    if show_entropy:
        plot.show()

if __name__ == '__main__':

    config = "config.csv"
    resources = "resources"
    results = "resources/beliefs"
    num_trials = 32
    show_entropy = True
    show_beliefs = True

    agents = {
        "BOPA": "orange",
        "ATPO": "green"
    }

    environments = [
        "gridworld",
        "pursuit-task",
        "pursuit-teammate",
        "pursuit-both",
        "abandoned_power_plant",
        "ntu",
        "overcooked",
        "isr",
        "mit",
        "pentagon",
        "cit"
    ]

    if show_beliefs:
        for environment in environments:
            plot_beliefs_environment(environment, agents, config, results, show_beliefs, show_entropy)
            print(flush=True)

    plot_mean_beliefs_environments(environments, agents, config, results, show_entropy)
