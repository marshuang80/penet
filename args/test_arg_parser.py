import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--cls_threshold', type=float, default=0.5,
                                 help='Threshold below which segmentation masks are set to 0.')
        self.parser.add_argument('--phase', type=str, default='val', choices=('train', 'val', 'test'),
                                 help='Phase to test on.')
        self.parser.add_argument('--results_dir', type=str, default='results/', help='Save dir for test results.')
        self.parser.add_argument('--visualize_all', type=util.str_to_bool, default=False,
                                 help='If true, visualize all examples.')
        self.parser.add_argument('--outputs_for_xgb', type=util.str_to_bool, default=False,
                                 help='If true, write output files for running XGBoost.')
        self.parser.add_argument('--save_segmentation', type=util.str_to_bool, default=False,
                                 help='If true, write segmentation outputs to disk.')
        self.parser.add_argument('--do_truncate', type=util.str_to_bool, default=False,
                                 help='If true, truncate predictions to just the head region.')
