import datetime
import random
import os.path as osp

import numpy as np

from algorithms.helpers import logger
from algorithms.helpers.file_logger import FileLogger


def rand_id(num_syllables=2, num_parts=3):
    """Randomly create a semi-pronounceable uuid"""
    part1 = ['s', 't', 'r', 'ch', 'b', 'c', 'w', 'z', 'h', 'k', 'p', 'ph', 'sh', 'f', 'fr']
    part2 = ['a', 'oo', 'ee', 'e', 'u', 'er']
    seps = ['_']  # [ '-', '_', '.']
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for i in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for i in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


class ExperimentInitializer:

    def __init__(self, args, name_prefix='', rank=None, world_size=None):
        """Initialize the experiment"""
        self.uuid = rand_id()
        self.args = args
        self.name_prefix = name_prefix
        self.rank = rank
        self.world_size = world_size
        # Set printing options
        np.set_printoptions(precision=3)

    def configure_logging(self):
        """Configure the experiment"""
        if self.rank is None:  # eval
            logger.info("configuring logger for evaluation")
            logger.configure(dir_=None, format_strs=['stdout'])

        elif self.rank == 0:  # train, master
            log_path = self.get_log_path()
            formats_strs = ['stdout', 'log', 'csv']
            fmtstr = "configuring logger"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.configure(dir_=log_path, format_strs=formats_strs)
            fmtstr = "logger configured"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.info("  directory: {}".format(log_path))
            logger.info("  output formats: {}".format(formats_strs))
            # In the same log folder, log args in yaml in yaml file
            file_logger = FileLogger(uuid=self.uuid,
                                     path=self.get_log_path(),
                                     file_prefix=self.name_prefix)
            file_logger.set_info('note', self.args.note)
            file_logger.set_info('uuid', self.uuid)
            file_logger.set_info('task', self.args.task)
            file_logger.set_info('args', str(self.args))
            fmtstr = "experiment configured"
            fmtstr += " [{} MPI workers]".format(self.world_size)
            logger.info(fmtstr)

        else:  # train, worker
            logger.info("configuring logger [worker #{}]".format(self.rank))
            logger.configure(dir_=None, format_strs=None)
            logger.set_level(logger.DISABLED)

    def prepend_date(undated_name):
        """Prepend the date to a file or directory name"""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S__{}".format(undated_name))

    def get_short_name(self):
        """Assemble short experiment name"""
        return "{}{}.".format(self.name_prefix, self.uuid)

    def get_long_name(self):
        """Assemble long experiment name"""
        name = self.get_short_name()
        if 'train' in self.args.task:
            name += "{}.".format(self.args.task)
        elif 'evaluate' in self.args.task:
            name += "{}.".format(self.args.task)
            assert self.args.num_trajs != np.inf, "num trajs must be finite"
            name += "num_trajs_{}.".format(self.args.num_trajs)
        else:
            raise NotImplementedError
        name += self.args.env_id.split('-')[0]  # get rid of the env version
        name += ".seed_{}".format(self.args.seed)
        return name

    def get_log_path(self):
        """Assemble the logging path"""
        if 'train' in self.args.task:
            log_name = self.get_long_name()
        else:
            raise NotImplementedError
        return osp.join(self.args.log_dir, log_name)
