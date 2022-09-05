import yaml
import os

def load_yaml(file_address):
    with open(file_address, 'r') as file:
        content = yaml.load_all(file, Loader = yaml.FullLoader)
        return list(content)

def dump_yaml(input_data, file_address):
    with open(file_address, 'w') as file:
        data = yaml.dump(input_data, file, default_flow_style = False, sort_keys = False)

def safe_load_yaml(file_address):
    with open(file_address, 'r') as file:
        try:
            config_settings = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
    return config_settings 

# define a function to save config file
# config label is the label of that configurations
def save_config(input_data, config_label, save_address):
    save_dic = {config_label: input_data}
    dump_yaml(save_dic, save_address)