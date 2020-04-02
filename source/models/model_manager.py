from abc import ABC, abstractmethod
import torch.nn as nn
import yaml

_templates = {}  # global registry of template classes

def template(cls):
    """
    This is a decorator for BaseUpscale-compliant template classes. Place
    `@base_model.template` on the line before a class defintion.

    This makes the class available to BaseUpscale (e.g. for reading saved steps from
    disk) whenever it's imported.

    """
    _templates[cls.__name__] = cls
    return cls

class AbstractUpScaler(ABC):
    """
    Abstract class representing an upscaler.
    """

    def __init__(self, net, device, upscale_factor):
        self.net = net.to(device)
        self.device = device
        self.upscale_factor = upscale_factor

    @abstractmethod
    def forward(self, input):
        """
        Return forward pass on input.

        Parameters
        ----------
        input: `torch.tensor`

        """
        pass

@template
class TemplateModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = 'template'
    
    @classmethod
    def from_config(cls, config_data):
        """
        create an input configuration from a saved yaml file
            
        Parameters
        ----------
        config_file: configuration file (yaml)
            
        Returns
        a upscaler moder
        """
        config_data.pop('name')
        return cls(**config_data)

    def forward(self, input):
        pass


class BaseScaler(AbstractUpScaler):
    """
    Wrap together the architecture -- net -- and add a train, forward and test methods

    Parameters
    ----------
    net: class
        model architecture inherited from TemplateModel

    """

    def __init__(self, net, device, upscale_factor):

        super().__init__(net, device, upscale_factor)

    @classmethod
    def from_config(cls, config_file):
        """
        Create a model input configuration from a saved yaml file

        Parameters
        ----------
        config_file : configuration file (yaml) path

        """

        with open(config_file, 'r') as stream:
            config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

        base_scaler = cls.from_dict(config_dict)

        return base_scaler

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a model input configuration from dictionary.

        Parameters
        ----------
        config_dict

        Returns
        -------

        """
        net_config = config_dict['net']  # dictionary defining the architecture
        net = _templates[net_config['name']].from_config(net_config)
        return net