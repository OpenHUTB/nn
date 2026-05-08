import click

import os.path

from what.utils.file import get_file

from what.cli.model import *
from what.cli.attack import *
from what.cli.example import *

# Main CLI (what)
@click.group()
def main_cli():
    """The CLI tool for WHite-box Adversarial Toolbox (WHAT)."""
    pass


# what model
@click.group()
def model():
    """Manage Deep Learning Models"""
    pass

# what model list
@model.command('list')
def model_list():
    """List supported models"""
    max_len = max([len(x[0]) for x in what_model_list])

    print('         {:<{w}s}\t{}\t{}'.format("       Model       ", "   Model Type   ", "Description", w=max_len))
    print('-' * 100)

    for i, model in enumerate(what_model_list, start=1):
        if os.path.isfile(os.path.join(WHAT_MODEL_PATH, model[WHAT_MODEL_FILE_INDEX])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[WHAT_MODEL_NAME_INDEX], model[WHAT_MODEL_TYPE_INDEX], model[WHAT_MODEL_DESC_INDEX], w=max_len))

# what model download
@model.command('download')
def model_download():
    """Download pre-trained models"""
    max_len = max([len(x[0]) for x in what_model_list])
 
    print('         {:<{w}s}\t{}\t{}'.format("       Model       ", "   Model Type   ", "Description", w=max_len))
    print('-' * 100)

    for i, model in enumerate(what_model_list, start=1):
        if os.path.isfile(os.path.join(WHAT_MODEL_PATH, model[WHAT_MODEL_FILE_INDEX])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[WHAT_MODEL_NAME_INDEX], model[WHAT_MODEL_TYPE_INDEX], model[WHAT_MODEL_DESC_INDEX], w=max_len))

    index = input(f"\nPlease input the model index: ")
    while not index.isdigit() or int(index) > len(what_model_list):
        index = input(f"Model [{index}] does not exist. Please try again: ")
    
    index = int(index) - 1
    get_file(what_model_list[index][WHAT_MODEL_FILE_INDEX],
             WHAT_MODEL_PATH,
             what_model_list[index][WHAT_MODEL_URL_INDEX],
             what_model_list[index][WHAT_MODEL_HASH_INDEX]
             )


# what attack
@click.group()
def attack():
    """Manage Attacks"""
    pass

# what attack list
@attack.command('list')
def attack_list():
    """List supported Attacks"""
    max_len = max([len(x[0]) for x in what_attack_list])

    for i, attack in enumerate(what_attack_list, start=1):
        print('{} : {:<{w}s}\t{}'.format(i, attack[WHAT_ATTACK_NAME_INDEX], attack[WHAT_ATTACK_TARGET_INDEX], w=max_len))


# what example
@click.group()
def example():  
    """Manage Examples"""
    pass

# what example list
@example.command('list')
def example_list():
    """List examples"""
    max_len = max([len(x[WHAT_EXAMPLE_NAME_INDEX]) for x in what_example_list])

    print('   {:<{w}s}\t{}\t{}'.format("        Demo        ", "       Type       ", "Description", w=max_len))
    print('-' * 80)

    for i, example in enumerate(what_example_list, start=1):
        print('{} : {:<{w}s}\t{}\t{}'.format(i, example[WHAT_EXAMPLE_NAME_INDEX], example[WHAT_EXAMPLE_TYPE_INDEX], example[WHAT_EXAMPLE_DESC_INDEX], w=max_len))

# what exmaple run
@example.command('run')
def example_run():
    """Run examples"""
    max_len = max([len(x[0]) for x in what_example_list])

    print('   {:<{w}s}\t{}\t{}'.format("        Demo        ", "       Type       ", "Description", w=max_len))
    print('-' * 80)

    for i, example in enumerate(what_example_list, start=1):
        print('{} : {:<{w}s}\t{}\t{}'.format(i, example[WHAT_EXAMPLE_NAME_INDEX], example[WHAT_EXAMPLE_TYPE_INDEX], example[WHAT_EXAMPLE_DESC_INDEX], w=max_len))

    index = input(f"\nPlease input the example index: ")
    while not index.isdigit() or int(index) > len(what_example_list):
        index = input(f"Example [{index}] does not exist. Please try again: ")

    what_example_list[int(index)-1][WHAT_EXAMPLE_FUNC_INDEX]()

def main():
    main_cli.add_command(model)
    main_cli.add_command(attack)
    main_cli.add_command(example)

    model.add_command(model_list)
    model.add_command(model_download)

    attack.add_command(attack_list)

    example.add_command(example_list)
    example.add_command(example_run)

    return main_cli()

if __name__ == "__main__":

    main()
