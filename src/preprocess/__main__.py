from src.preprocess.audio import preprocess_trainset, parse_args


if __name__ == "__main__":
    preprocess_trainset(*parse_args())
