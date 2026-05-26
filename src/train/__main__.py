import torch.multiprocessing as mp

from src.train.runner import main


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
