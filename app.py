from src.data_loader import load_data
from src.model_trainer import model_train


if __name__ == "__main__":
    yaml_file_path = r"C:\Users\LokeshKanna\Downloads\ClearML\params.yaml"

    load_data(yaml_file_path)

    model_train(yaml_file_path)
