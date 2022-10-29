import argparse, os
import save_test_submission

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = root_parser.add_subparsers(title='subcommands', description='subcommands implemented',
                                        help='choose specific subcommand for additional help')


#--------------------------------------------------Command verbs (sub-commands) registration-------------------------------------------#
"""
function for save_test_submission/get_test_predictions
"""
##common arguments
parser = subparsers.add_parser('save_test_submission', help='eval the prediction',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", type=str, help="gpu id, e.g., '0,1,2,3'")
parser.add_argument("--aml", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether to use AML")
parser.set_defaults(func=save_test_submission.get_test_predictions)

parser.add_argument("--path", type=str, default="../model_output", help="model output")
parser.add_argument('--cand_path', type=str, default='cand_data', help="indicate the candidate path")
parser.add_argument('--num_proc', type=int, default=1,
                        help='The number of processes to evaluate the model in parallel.'\
                                'For multi-GPU, the number of processes by default is set to match the number of GPUs.'\
                                'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
parser.add_argument("--with_test", type=lambda x: (str(x).lower() == 'true'), default=True, help="whether to use test")

parser.add_argument('--model_prefix', type=str, default='OTE_wikikg90m_concat_d_240_g_12.00')

"""
function for xxx
"""


#--------------------------------------------------Command verbs (sub-commands) registration-------------------------------------------#
args = root_parser.parse_args()
args.func(args)