
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('echo', None, 'Text to echo.')
flags.DEFINE_integer('epochs', 300, 'Number of Epochs')
flags.DEFINE_bool('start_train', False, 'Activates Training')

def main(argv):
  del argv  # Unused.

  print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
        file=sys.stderr)
  logging.info('echo is %s.', FLAGS.echo)
  logging.info('epochs is %s.', FLAGS.epochs)
  logging.info('start_train is %s.', FLAGS.start_train)


if __name__ == '__main__':
  app.run(main)