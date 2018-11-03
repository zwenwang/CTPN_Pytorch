import ConfigParser


if __name__ == '__main__':
    cp = ConfigParser.ConfigParser()
    cp['global'] = {
        'using_cuda': 'True',
        'epoch': '12',
        'gpu_id': '0',
        'display_file_name': 'False',
        'display_iter': '10',
        'val_iter': '30',
    }
    with open('./config', 'w') as fp:
        cp.write(fp)
