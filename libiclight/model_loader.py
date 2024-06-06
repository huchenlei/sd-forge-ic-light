from enum import Enum
import os

FC_NAME: str = None
FBC_NAME: str = None
FC_PATH: str = None
FBC_PATH: str = None


def detect_models(models_path: str):
    global FC_NAME, FBC_NAME, FC_PATH, FBC_PATH
    folder = os.path.join(models_path, "ic-light")

    if not os.path.exists(folder):
        print('\n[Warning] "ic-light" folder is not detected in the models folder!')
        print("Create the folder and download the models from Releases!\n")

    else:
        for obj in os.listdir(folder):
            if "fc" in obj.lower():
                FC_NAME = os.path.splitext(obj)[0]
                FC_PATH = os.path.join(folder, obj)
            elif "fbc" in obj.lower():
                FBC_NAME = os.path.splitext(obj)[0]
                FBC_PATH = os.path.join(folder, obj)

        if FC_PATH is None:
            print('\n[Warning] "FC" model not detected!\nDownload it from Releases!')
        if FBC_PATH is None:
            print('\n[Warning] "FBC" model not detected!\nDownload it from Releases!')


class ModelType(Enum):
    FCON = -1
    FC = 0
    FBC = 1

    @classmethod
    def get(cls, value: str):
        global FC_NAME, FBC_NAME

        if value == FC_NAME:
            return ModelType.FC
        elif value == FBC_NAME:
            return ModelType.FBC
        else:
            raise SystemError

    @property
    def name(self) -> str:
        global FC_NAME, FBC_NAME

        match (self):
            case ModelType.FC:
                return FC_NAME
            case ModelType.FBC:
                return FBC_NAME
            case _:
                raise SystemError

    @property
    def path(self) -> str:
        global FC_PATH, FBC_PATH

        match (self):
            case ModelType.FC:
                return FC_PATH
            case ModelType.FBC:
                return FBC_PATH
            case _:
                raise SystemError
