import os
from fire import Fire


def main(
    num
):
    num=str(num)
    #dataset = "ml-1m_pivot3"
    seeds = {"0": 273, "1": 588, "2": 600, "3": 681, "4": 846}
    device = {"0": "cuda:0", "1": "cuda:0", "2": "cuda:0", "3": "cuda:0", "4": "cuda:0"}
    lr = [1e-3]
    gnn_layers = [2] 
    ds = [16, 32]
    lambda1s = [1e-5, 1e-6, 1e-7]
    lambda2s = [1e-4, 1e-5]
    temps = [0.3, 0.5, 1, 3, 10]
    dropouts = [0, 0.25]
    qs = [5]
    input_dim = [16, 32]

    use_mlflow = True
    for dataset in ["digi_music"]:
        for indim in input_dim: 
            os.system(f"python main.py --data {dataset} --seed {seeds[num]} --cuda {device[num]} --d {indim}")


if __name__ == "__main__":
    Fire(main)
