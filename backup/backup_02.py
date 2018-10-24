import os
import time

source = ['"C:\\Only for men"', 'C:\\My_docs']

target_dir = 'E:\\Backup'

extension = '.zip'

subDirectory = target_dir + os.sep + time.strftime('%Y%m%d')

now = time.strftime('%H%M%S')

target = subDirectory + os.sep + now
comment = input('Введите комментарий --> ')
if len(comment) == 0:
    target += extension
else:
    target += '_' + comment.replace(' ', '_') + extension

if not os.path.exists(subDirectory):
    os.mkdir(subDirectory)

zip_command = "zip -qr {0} {1}".format(target, ' '.join(source))

if os.system(zip_command) == 0:
    print('Резервная копия успешно создана в ', target)
else:
    print('Создание резервной копии не удалось')
