class RequiredArgumentLost(Exception):
    """
    Забыли указать параметр, обязательный при других параметрах
    """
    def __init__(self, key_arg, key_value, required_args, lost_args):
        self.text = "Параметр {} = {} требует следующих параметров: {}\nПожалуйста, укажите {}"\
            .format(key_arg, key_value, ', '.join(required_args), ', '.join(lost_args))

    def __str__(self):
        return self.text


def check_args(args, key_arg, required_dict):
    key_value = args.__getattribute__(key_arg)
    if key_value in required_dict:
        required_args = required_dict[key_value]
        # собираем обязательные аргументы, которых нет
        lost_args = {arg for arg in required_args if not args.__getattribute__(arg)}
        if lost_args:
            raise RequiredArgumentLost(key_arg, key_value, required_args, lost_args)


def arg_to_list(arg, splitter = ','):
    if isinstance(arg, list):
        return arg
    else:
        new_arg = [int(x) for x in arg.split(splitter)]
        return new_arg
