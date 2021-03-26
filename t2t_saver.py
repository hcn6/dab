"""Decode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.models import transformer
import problems
import tensorflow as tf
import decoding
import t2t_decoder
import t2t_trainer
from tensor2tensor.utils import registry

flags = tf.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(t2t_decoder.main)

  hp, decode_hp, estimator = decoding.create_hp_and_estimator(
      FLAGS.problem, FLAGS.data_dir, FLAGS.checkpoint_path or FLAGS.output_dir)
  
  model_cls = t2t_decoder.registry.model(FLAGS.model)

  run_config = t2t_decoder.t2t_trainer.create_run_config(hp)

  model = model_cls(
      hp, 
      tf.estimator.ModeKeys.EVAL,
      data_parallelism=None,
      decode_hparams=decode_hp
  )

  model.get_scaffold_fn(FLAGS.checkpoint_path or FLAGS.output_dir)()

  model.save('Translation_Vien_Base1m')
