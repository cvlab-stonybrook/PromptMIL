import time


def add_common_arguments(parser):

    # dataset related parameters
    parser.add_argument("dataset_root", type=str, help='The path of hdf5 file.')
    parser.add_argument("dataset_csv", type=str, help='The csv of dataset.')
    parser.add_argument("--output-dir", type=str, help="An output directory")

    parser.add_argument("--val_fold", type=int, default=0, help="Which fold is used for evaluation")

    parser.add_argument("--batch-size-train", type=int, default=128, help="Choose the batch size for training the "
                                                                         "backbone network")
    parser.add_argument("--batch-size-eval", type=int, default=1024, help="Choose the batch size for evaluating the "
                                                                          "backbone network")

    parser.add_argument("--dataset-name", type=str, default=None, help="The name of input dataset. "
                                                                       "[tcga-luadsc|tcga-brca|tcga-paad|bright]")
    parser.add_argument('--data-norm', action="store_true", help='Whether to normlize data using data-mean, data-std')
    parser.add_argument('--data-mean', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='mean of the dataset')
    parser.add_argument('--data-std', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='std of the dataset')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers to use for data loading')

    # model and network realted parameters
    parser.add_argument("--model", type=str, default=None, help="type of MIL model, eg. dsmil")
    parser.add_argument("--transfer-type", type=str, default=None, help="type transfer learning, eg. prompt")

    parser.add_argument("--network", type=str, default=None, help="type of backbone network, eg. vit_tiny_patch16_384")

    parser.add_argument("--num-prompt-tokens", type=int, default=1, help="number of prompt tokens")
    parser.add_argument('--deep-prompt', action="store_true", help='')
    parser.add_argument("--prompt-dropout", type=float, default=0., help="")
    parser.add_argument("--project-prompt-dim", type=int, default=-1, help="")

    # Learning rate schedule parameters
    parser.add_argument("--epochs", type=int, default=40, help="How many epochs to train for")

    parser.add_argument("--precision", type=int, default=32, help="32 or 16 bit precision training")

    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='Weight decay of the optimizer (default: 1e-2)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-factor', type=float, default=1.,
                        help='learning rate multiplication for pretrained networks (default: 1.)')

    parser.add_argument('--adam', action="store_true",
                        help='Use Adam optimizer if set to true, otherwise use AdamW.')
    parser.add_argument('--loss-weight', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='Weight of each class')
    parser.add_argument('--auto-loss-weight', action="store_true", help='Automatically calculate the weight of each class')

    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='simulate larger batch size by '
                                                                               'accumulating gradients')
    # Dropout
    parser.add_argument('--dropout-inst', type=float, default=0.0, help='Dropout rate for patches')
    parser.add_argument('--dropout-att', type=float, default=0., help='Dropout rate for attentions')

    # pretrained weights related
    parser.add_argument('--pretrained', action="store_true", help='load imagenet pretrained weight')
    parser.add_argument('--load-backbone-weight', type=str, default=None, help='If not None, load weights from given path')
    parser.add_argument('--load-weights', type=str, default=None, help='If not None, load weights from given path')

    # gpu
    parser.add_argument('--gpu-id', type=lambda s: [int(item) for item in s.split(',')], default=None)

    # project name and tag
    parser.add_argument('--run-name', type=str, default='test')
    parser.add_argument('--tag', type=str, default='', help="For logging only")


    return parser


def get_arguments(parser):
    parser = add_common_arguments(parser)
    opts = parser.parse_args()
    opts = process_common_arguments(opts)
    return opts


def process_common_arguments(opts):
    # opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    return opts

def get_arguments_additional(parser, add_argument_fun, process_argument_fun):
    parser = add_common_arguments(parser)
    if add_argument_fun is not None:
        parser = add_argument_fun(parser)
    opts = parser.parse_args()
    opts = process_common_arguments(opts)
    if process_argument_fun is not None:
        opts = process_argument_fun(opts)
    return opts
