import os
import random

# variables
KB = 1024
MB = KB * 1024
GB = MB * 1024

if not os.path.isfile('../data/before_compress/mb64'):
  with open('../data/before_compress/mb64', 'wb') as fout:
    fout.write(os.urandom(64 * MB))
    print("64")

if not os.path.isfile('../data/before_compress/mb128'):
  with open('../data/before_compress/mb128', 'wb') as fout:
    fout.write(os.urandom(128 * MB))
    print("128")

if not os.path.isfile('../data/before_compress/mb256'):
  with open('../data/before_compress/mb256', 'wb') as fout:
    fout.write(os.urandom(256 * MB))
    print("256")

if not os.path.isfile('../data/before_compress/mb512'):
  with open('../data/before_compress/mb512', 'wb') as fout:
    fout.write(os.urandom(512 * MB))
    print("512")
	
# if not os.path.isfile('mb768'):
#   with open('mb768', 'wb') as fout:
#     fout.write(os.urandom(786 * MB))
#     print("786")

# if not os.path.isfile('mb1024'):
#   with open('mb1024', 'wb') as fout:
#     fout.write(os.urandom(1024 * MB))
#     print("1024")

# if not os.path.isfile('mb1280'):
#   with open('mb1280', 'wb') as fout:
#     fout.write(os.urandom(1280 * MB))
#     print("1280")

# if not os.path.isfile('mb1536'):
#   with open('mb1536', 'wb') as fout:
#     fout.write(os.urandom(1536 * MB))
#     print("1536")
	
# if not os.path.isfile('mb1792'):
#   with open('mb1792', 'wb') as fout:
#     fout.write(os.urandom(1792 * MB))
#     print("1792")

# if not os.path.isfile('mb2048'):
#   with open('mb2048', 'wb') as fout:
#     fout.write(os.urandom(2048 * MB))
#     print("2048")

# if not os.path.isfile('mb2304'):
#   with open('mb2304', 'wb') as fout:
#     fout.write(os.urandom(2304 * MB))
#     print("2304")

