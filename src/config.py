import typing as t
import tensorflow as tf

from dataclasses import asdict, dataclass, field, fields
from dict_to_dataclass import DataclassFromDict, field_from_dict


@dataclass
class Config(DataclassFromDict):
    learning_rate: float = field_from_dict("learning_rate", default=0.001)
    epochs: int = field_from_dict("epochs", default=100)
    batch_size: int = field_from_dict("batch_size", default=8)
    loss_function: t.Union[t.Callable, str] = field_from_dict("loss_function", default=tf.keras.losses.CategoricalCrossentropy())
    
    def __post_init__(self):
        if isinstance(self.loss_function, str):
            self.loss_function = tf.keras.losses.get(self.loss_function)
            if not isinstance(self.loss_function, tf.keras.losses.Loss):
                raise ValueError(
                    "Invalid string given for loss function class"
                )

    @classmethod
    def instantiate(cls, classToInstantiate, argDict):
        if classToInstantiate not in cls.classFieldCache:
            cls.classFieldCache[classToInstantiate] = {
                f.name for f in fields(classToInstantiate) if f.init
            }

        fieldSet = cls.classFieldCache[classToInstantiate]
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return classToInstantiate(**filteredArgDict)

    def to_dict(self):
        return {
            k: v.name if isinstance(v, t.Callable) else repr(v)
            for k, v in asdict(self).items()
        }
