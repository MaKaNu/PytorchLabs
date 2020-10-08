
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags

import logging

from SemanticSegmentation.utils.logger import CustomFormatter as CF

FLAGS = flags.FLAGS

flags.DEFINE_string('echo', None, 'Text to echo.')

def main(argv):
    del argv  # Unused.

    logger = logging.getLogger("test_gflags.py")
    logger.setLevel(logging.INFO)

    # create console handler with a higher log level
    consoleh = logging.StreamHandler()
    consoleh.setLevel(logging.INFO)

    consoleh.setFormatter(CF())

    logger.addHandler(consoleh)

    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
        file=sys.stderr)
    logger.info('echo is %s.', FLAGS.echo)

if __name__ == '__main__':
    app.run(main)
