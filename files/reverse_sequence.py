import numpy as np
def reverse_sequence(inputFilename):
    mode = "r+"
    file = open(inputFilename, mode)
    lines = file.readlines()
    file.seek(0, 0)
    for sample in lines:
        elements = sample.split(sep=' ')
        for el in elements:
            if el.endswith('\n'):
                file.write(el)
            else:
                file.write(el+'\n')
    file.close()
