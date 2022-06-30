import os
import yaml

# ABINet 참조 
class load_config(object):

    def __init__(self, config_path, host=True):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_')
                else:
                    if k == 'model_name':
                        assert v in ['LeNet5', 'AlexNet', 
                                      'VGGNetA','VGGNetA-LRN','VGGNetB','VGGNetC','VGGNetD','VGGNetE']
                        
                    self.__setattr__(f'{prefix}{k}', v)
        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        __dict2attr(config_dict)

    def __getattr__(self, item):
        attr = self.__dict__.get(item)
        if attr is None:
            attr = dict()
            prefix = f'{item}_'
            for k, v in self.__dict__.items():
                if k.startswith(prefix):
                    n = k.replace(prefix, '')
                    attr[n] = v
            return attr if len(attr) > 0 else None
        else:
            return attr

    def __repr__(self):
        str = 'ModelConfig(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str