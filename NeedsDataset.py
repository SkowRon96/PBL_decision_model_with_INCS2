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

    if 'LOW' in args.hierarchy:
        log.info("Creating Data Frame for low level needs hierarchy...")
        d = {'food': [r.uniform(0, 1)], 'drinking': [r.uniform(0, 1)], 'dream': [r.uniform(0, 1)],
             'sex': [r.uniform(0.3, 1)], 'toilet': [r.uniform(0, 1)]}
        d_out = {'output_need': ["NONE"]}
        df = pd.DataFrame(data=d)
        df_out = pd.DataFrame(data=d_out)
        for i in range(args.number_samples - 1):
            d1 = {'food': [r.uniform(0, 1)],
                  'drinking': [r.uniform(0, 1)],
                  'dream': [r.uniform(0, 1)],
                  'sex': [r.uniform(0.3, 1)],
                  'toilet': [r.uniform(0, 1)]}
            d1_out = {'output_need': ['NONE']}
            df1 = pd.DataFrame(data=d1)
            df = pd.concat([df, df1])
            df1_out = pd.DataFrame(data=d1_out)
            df_out = pd.concat([df_out, df1_out])

    if 'HIGH' in args.hierarchy:
        log.info("Creating Data Frame for low and high level needs hierarchy...")
        d = {'food': [r.uniform(0, 1)], 'drinking': [r.uniform(0, 1)], 'dream': [r.uniform(0, 1)],
             'sex': [r.uniform(0.3, 1)], 'toilet': [r.uniform(0, 1)], 'safety': [r.uniform(0.3, 1)],
             'love&membership': [r.uniform(0.3, 1)],'respect&recognition': [r.uniform(0.3, 1)],
             'self_realization': [r.uniform(0.3, 1)]}
        d_out = {'output_need': ["NONE"]}
        df = pd.DataFrame(data=d)
        df_out = pd.DataFrame(data=d_out)
        for i in range(args.number_samples - 1):
            d1 = {'food': [r.uniform(0, 1)],
                  'drinking': [r.uniform(0, 1)],
                  'dream': [r.uniform(0, 1)],
                  'sex': [r.uniform(0.3, 1)],
                  'toilet': [r.uniform(0, 1)],
                  'safety': [r.uniform(0.3, 1)],
                  'love&membership': [r.uniform(0.3, 1)],
                  'respect&recognition': [r.uniform(0.3, 1)],
                  'self_realization': [r.uniform(0.3, 1)]}
            d1_out = {'output_need': ['NONE']}
            df1 = pd.DataFrame(data=d1)
            df = pd.concat([df, df1])
            df1_out = pd.DataFrame(data=d1_out)
            df_out = pd.concat([df_out, df1_out])

    df.index = range(args.number_samples)
    df_out.index = range(args.number_samples)
    #print(df.head())

    if 'YES' in args.decision_making:
        log.info("Decision-making process. Generating output needs...")

        if 'HIGH' in args.hierarchy:
            dfh = df.iloc[:,5:9]
            dfh_min_ind = dfh.idxmin(axis=1)

        df = df.iloc[:, 0:5]
        df_min_ind = df.idxmin(axis=1)
        # Determining output for low levels needs below 0.3
        for i in range(len(df)):
            if df.at[i, df_min_ind.loc[i]] < 0.3:
                if df_min_ind.loc[i] == 'food':
                    df_out.at[i, 'output_need'] = 'go_eat'
                if df_min_ind.loc[i] == 'drinking':
                    df_out.at[i, 'output_need'] = 'go_drink'
                if df_min_ind.loc[i] == 'toilet':
                    df_out.at[i, 'output_need'] = 'go_to_toilet'
                if df_min_ind.loc[i] == 'sex':
                    df_out.at[i, 'output_need'] = 'go_sex'
                if df_min_ind.loc[i] == 'dream':
                    df_out.at[i, 'output_need'] = 'go_sleep'


        # Determining output for low levels needs above 0.3 and below 0.8
        # Determining list of needs with assumption minimum(need[i])-need[i]<0.1
        for i in range(len(df)):
            possible_needs = list()
            for j in range(len(df.columns)):
                if df.at[i, df_min_ind.loc[i]] > 0.3 and df.at[i, df_min_ind.loc[i]] < 0.8:
                    if (df.iat[i, j] - df.at[i, df_min_ind.loc[i]]) < 0.1:
                        possible_needs.append(df.columns[j])

            if len(possible_needs) == 1:
                num1 = 0
                if possible_needs[num1] == 'food':
                    df_out.at[i, 'output_need'] = 'go_eat'
                if possible_needs[num1] == 'drinking':
                    df_out.at[i, 'output_need'] = 'go_drink'
                if possible_needs[num1] == 'toilet':
                    df_out.at[i, 'output_need'] = 'go_to_toilet'
                if possible_needs[num1] == 'sex':
                    df_out.at[i, 'output_need'] = 'go_sex'
                if possible_needs[num1] == 'dream':
                    df_out.at[i, 'output_need'] = 'go_sleep'
            if len(possible_needs) > 1:
                num1 = random.randint(0, len(possible_needs) - 1)
                if possible_needs[num1] == 'food':
                    df_out.at[i, 'output_need'] = 'go_eat'
                if possible_needs[num1] == 'drinking':
                    df_out.at[i, 'output_need'] = 'go_drink'
                if possible_needs[num1] == 'toilet':
                    df_out.at[i, 'output_need'] = 'go_to_toilet'
                if possible_needs[num1] == 'sex':
                    df_out.at[i, 'output_need'] = 'go_sex'
                if possible_needs[num1] == 'dream':
                    df_out.at[i, 'output_need'] = 'go_sleep'

        for i in range(len(df)):
            if df.at[i, df_min_ind.loc[i]] > 0.8 and 'HIGH' in args.hierarchy:
                low_high_flag = random.randint(0, 1)
            else:
                low_high_flag=0

            #if low_high_flag==1:
                #realization of high level needs

            if df.at[i, df_min_ind.loc[i]] > 0.8 and low_high_flag==0:
                draw=random.randint(0, len(df.columns) - 1)

                if df.iat[i, draw]  == 'food':
                   df_out.at[i, 'output_need'] = 'go_eat'
                if df.iat[i, draw] == 'drinking':
                   df_out.at[i, 'output_need'] = 'go_drink'
                if df.iat[i, draw] == 'toilet':
                    df_out.at[i, 'output_need'] = 'go_to_toilet'
                if df.iat[i, draw] == 'sex':
                    df_out.at[i, 'output_need'] = 'go_sex'
                if df.iat[i, draw] == 'dream':
                    df_out.at[i, 'output_need'] = 'go_sleep'

    if 'HIGH' in args.hierarchy:
        df = pd.concat([df, dfh, df_out], axis=1, sort=False)
    else:
        df = pd.concat([df, df_out], axis=1, sort=False)
        #df = pd.concat([df], axis=1, sort=False)
        print(df.head())

    df.to_csv('needs.csv', index=False)
    log.info("End - output file needs.csv")


if __name__ == '__main__':
    sys.exit(main() or 0)
