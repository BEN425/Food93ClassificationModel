'''
cfgparser.py

Read config file (yaml format)
Remove argparse features to fix error in Jupyter notebook
'''


import yaml
import argparse

class CfgParser():
    def __init__(self, config_path):
        config = self._load_config(config_path)
        # args = self._parser(config).__dict__
        # self.cfg_dict = self._merge_args2yml(args, config)
        
        # Combine settings
        self.cfg_dict = {**config["PARSER_SETTING"], **config["TRAIN_SETTING"]}
        
    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    
    def _parser(self, config):
        setting = config["PARSER_SETTING"]
        parser = argparse.ArgumentParser()
        parser.add_argument("--SEED", type=int, default=setting["SEED"])
        parser.add_argument("--GPU_ID", type=str, default=setting["GPU_ID"])
        parser.add_argument("--WORKERS", type=int, default=setting["WORKERS"])
        parser.add_argument("--EPOCHS", type=int, default=setting["EPOCHS"])
        parser.add_argument("--BATCH_SIZE", type=int, default=setting["BATCH_SIZE"])
        parser.add_argument("--EVAL_BATCH_SIZE", type=int, default=setting["EVAL_BATCH_SIZE"]) 
        args = parser.parse_args()
        return args
    
    def _merge_args2yml(self, args, config):
        for key, value in args.items():
            config["PARSER_SETTING"][key] = value
        merge_dict = {**config["PARSER_SETTING"], **config["TRAIN_SETTING"]}
        return merge_dict

if __name__ == "__main__":
    cfgparser = CfgParser(config_path="./cfg/CombinationFood.yml")
    data = cfgparser.cfg_dict
    print(data)
