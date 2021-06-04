import sys
from argparse import ArgumentParser, SUPPRESS
import logging as log
import pandas as pd
import random as r


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-nh", "--hierarchy", help="Optional. Determination of level of needs hierarchy", default="LOW",
                      type=str)
    args.add_argument("-ns", "--number_samples", help="Required. Number of output samples", required=True, type=int)
    args.add_argument("-dm", "--decision_making", help="Optional. Generating output needs.", default="NO", type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    df = pd.DataFrame()

    if 'HIGH' in args.hierarchy:
        log.info("Creating Data Frame for low and high level needs hierarchy...")
        d = {'food': [r.uniform(0, 1)], 'drinking': [r.uniform(0, 1)], 'dream': [r.uniform(0, 1)],
             'sex': [r.uniform(0.3, 1)], 'toilet': [r.uniform(0, 1)], 'high': [r.uniform(0.3, 1)]}
        d_out = {'output_need': ["NONE"]}
        df = pd.DataFrame(data=d)
        df_out = pd.DataFrame(data=d_out)
        for i in range(args.number_samples - 1):
            d1 = {'food': [r.uniform(0, 1)],
                  'drinking': [r.uniform(0, 1)],
                  'dream': [r.uniform(0, 1)],
                  'sex': [r.uniform(0.3, 1)],
                  'toilet': [r.uniform(0, 1)],
                  'high': [r.uniform(0.3, 1)]}
            d1_out = {'output_need': [r.randint(0, 5)]}
            df1 = pd.DataFrame(data=d1)
            df = pd.concat([df, df1])
            df1_out = pd.DataFrame(data=d1_out)
            df_out = pd.concat([df_out, df1_out])

    df.index = range(args.number_samples)
    df_out.index = range(args.number_samples)

    for i in range(len(df_out)):
        if df_out.at[i, 'output_need'] == 0:
            df_out.at[i, 'output_need']= 'a_go_eat'
        if df_out.at[i, 'output_need'] == 1:
            df_out.at[i, 'output_need'] = 'b_go_drink'
        if df_out.at[i, 'output_need'] == 2:
            df_out.at[i, 'output_need'] = 'c_go_sleep'
        if df_out.at[i, 'output_need'] == 3:
            df_out.at[i, 'output_need'] = 'd_go_sex'
        if df_out.at[i, 'output_need'] == 4:
            df_out.at[i, 'output_need'] = 'e_go_to_toilet'
        if df_out.at[i, 'output_need'] == 5:
            df_out.at[i, 'output_need'] = 'f_go_high'


    df = pd.concat([df,df_out], axis=1, sort=False)
    df.to_csv('needs2.csv', index=False)
    log.info("End - output file needs.csv")


if __name__ == '__main__':
    sys.exit(main() or 0)
