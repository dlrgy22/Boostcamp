from dataclasses import dataclass

@dataclass
class Config:
    device: str
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    IMG_SIZE:int
    MODEL_NAME:str
    PRETRAINED:bool