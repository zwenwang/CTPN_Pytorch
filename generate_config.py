import ConfigParser


if __name__ == '__main__':
    cp = ConfigParser.ConfigParser()
    cp.add_section('global')
    cp.set('global', 'using_cuda', 'False')
    cp.set('global', 'epoch', '12')
    cp.set('global', 'gpu_id', '0')
    cp.set('global', 'display_file_name', 'False')
    cp.set('global', 'display_iter', '10')
    cp.set('global', 'val_iter', '30')
    with open('./config', 'w') as fp:
        cp.write(fp)
