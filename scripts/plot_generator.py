import matplotlib.pyplot as plt
import json

colors = {
    "DQN" : "red",
    "PPO" : "green",
    "APPO" : "blue",
    "IMPALA" : "orange"
}

def plot_rewards(scenario=None, episode_rewards=None, smoothing=None,
                 filename=None, color_map=colors):
    print(f"Showing {scenario}.")
    # print(f"all episode rewards from all algorithms: {episode_rewards}")
    for algo in episode_rewards:
        color = "purple" if algo not in color_map else color_map[algo]

        # no smoothing
        if smoothing is None or smoothing.lower() == "no_smoothing":
            y_values = episode_rewards[algo]
            x_values = list(range(len(y_values)))
            plt.plot(x_values, y_values, label=algo, color=color)

        # only includes datapoints from every 5 episodes
        elif smoothing.lower() == "downsampling":
            y_values = episode_rewards[algo]
            smoothed_y_values = []
            for i in range(0, len(y_values), 5):
                smoothed_y_values.append(y_values[i])
            x_values = list(range(len(smoothed_y_values)))
            for i in range(len(x_values)): x_values[i] *= 5
            plt.plot(x_values, smoothed_y_values, label=algo, color=color)

        # uses an efficient version of the moving average function in O(n)
        elif smoothing.lower() == "window_moving_average":
            y_values = episode_rewards[algo]
            smoothed_y_values = []
            sum = 0.0
            window_len = 10
            for i in range(len(y_values)):
                sum += y_values[i]
                if i >= window_len:
                    sum -= y_values[i-window_len]
                    avg = float(sum) / window_len
                    smoothed_y_values.append(avg)
                else:
                    smoothed_y_values.append(y_values[i])
            x_values = list(range(len(smoothed_y_values)))
            plt.plot(x_values, smoothed_y_values, label=algo, color=color)

        # uses the exponentially weighted moving average function
        elif smoothing.lower() == "ewma":
            y_values = episode_rewards[algo]
            smoothed_y_values = []
            avg = 0.0
            beta = 0.75
            for i in range(len(y_values)):
                avg = beta * avg + (1 - beta) * y_values[i]
                smoothed_y_values.append(avg)
            x_values = list(range(len(smoothed_y_values)))
            plt.plot(x_values, smoothed_y_values, label=algo, color=color)

    plt.xlabel("Episode Number")
    plt.ylabel("Episode Reward")
    plt.title(f"{scenario}: Episode Reward vs. Episodes Elapsed")
    plt.legend()

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if filename is not None:
        plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    dict_path = "results_from_paper.json"
    with open(dict_path, "r") as file:
        results_dict = json.load(file)

    for scenario in results_dict:
        plot_rewards(
            scenario=scenario,
            episode_rewards=results_dict[scenario]["episode_rewards"],
            smoothing="ewma",
        )
