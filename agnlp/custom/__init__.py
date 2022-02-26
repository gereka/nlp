
# These imports are important for making the configuration files find the classes that you wrote.
# If you don't have these, you'll get errors about allennlp not being able to find
# "simple_classifier", or whatever name you registered your model with.  These imports and the
# contents of .allennlp_plugins makes it so you can just use `allennlp train`, and we will find your
# classes and use them.  If you change the name of `my_project`, you'll also need to change it in
# the same way in the .allennlp_plugins file.
#from customt.model import *
from agnlp.custom.dataset_reader import *
from agnlp.custom.commands import *
from agnlp.custom.modules.seq2seq_encoders import *
print('hello from agnlp.custom')
