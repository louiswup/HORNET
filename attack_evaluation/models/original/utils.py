from .stutz2020.ccat import load_ccat_model
from .wang2023.dm_adv_training import load_dm_adv_model
from .xiao2020.kwta import load_kwta_model
from .zhang2020.crown import load_crown_model
from .cohen19.random_smooth import load_rand_smooth_model
from .yang19.MENet import load_MENet_model

_available_defenses = {'Stutz2020CCAT': load_ccat_model,
                       'Zhang2020CrownLarge': load_crown_model,
                       'Zhang2020CrownSmall': load_crown_model,
                       'Xiao2020KWTA': load_kwta_model,
                       'Wang2023DMAdvSmall': load_dm_adv_model,
                       'Wang2023DMAdvLarge': load_dm_adv_model,
                       'Cohen2019Certified': load_rand_smooth_model,
                       'Yang2019MENet': load_MENet_model}


def load_original_model(model_name: str, dataset: str, threat_model: str):
    return _available_defenses[model_name](model=model_name, dataset=dataset, threat_model=threat_model)
