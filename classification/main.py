from classification.configure import Config
from classification.train_bert import train


def main():
    config = Config()

    print("使用设备: ", config.device)

    # 正式开始训练
    train(config)


if __name__ == "__main__":
    main()
