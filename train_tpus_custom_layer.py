import os
import shlex
from time import sleep
from multiprocessing import Process
import subprocess

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('index', 'envi' , 'task to train')
flags.DEFINE_integer('index', 0, 'index to train')
# os.system("pip install google-colab")
from google.colab import auth
auth.authenticate_user()
print('authenticated')


TPU_ADDRESSES = '34.134.112.85'

task = 'envi'
total_train_steps = 500000
use_tpu = True
  # TPU wants the address to begin with gs://
train_output_dir = f'gs://best_vi_translation/checkpoints/seq_length_translate_class11_pure_envi_iwslt32k/len_512'
train_data_dir = f'gs://best_vi_translation/data/translate_class11_pure_envi_iwslt32k/'
# print(train_output_dir)
# print(train_data_dir)
hparams_str = ('learning_rate_cosine_cycle_steps={},'
               'max_length=512,batch_size=4096,'  # real batch_size = 4096/128
               'learning_rate_constant=2.0').format(2000000)
print(hparams_str)

hparams_set = 'transformer_tall9'
problem = 'translate_class11_pure_envi_iwslt32k'
model = 'transformer'
string = f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model={model} --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu}"
os.system(string)
