import yaml

def get_config(config_file_path: str):
    with open(config_file_path, 'r') as yml_file:
        consts = yaml.safe_load(yml_file)
    return consts
