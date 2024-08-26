import pandas as pd
import numpy as np
import csv 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.use('pdf')

def read_data(summary_file, subsample=4):
    df = pd.read_csv(summary_file)
    iterations = np.array(df["Iteration"])[0::subsample]*subsample
    QD_Score = np.array(df["QD-Score"])[0::subsample]
    Coverage = np.array(df["Coverage"])[0::subsample]*100
    BestReward = np.array(df["Maximum"])[0::subsample]
    AvgReward = np.array(df["Average"])[0::subsample]
    # print(iterations.shape)
    # print(QD_Score.shape)
    # print(Coverage)
    # print(BestReward)
    # print(AvgReward.shape)
    return iterations, QD_Score, Coverage, BestReward, AvgReward

def get_resultsfile(resultsfolder='experiments',method='gail',game='ant',seed='1111'):
    return f'{resultsfolder}/IL_ppga_{game}_{method}/{seed}/summary.csv'

def get_score_from_resultsfile(resultsfolder,method,game,seed):
    filename = get_resultsfile(resultsfolder,method,game,seed)
    iterations, QD_Score, Coverage, BestReward, AvgReward = read_data(filename)
    return iterations, QD_Score, Coverage, BestReward, AvgReward

def get_method_scores(resultsfolder,game,methods,labels,seeds):
    evals=2000
    qd_scores = {}
    times = {}
    coverages = {}
    best_perf = {}
    avg_perf = {}
    for i, method in enumerate(methods):
        times[labels[i]] = np.zeros((len(seeds),evals))
        qd_scores[labels[i]] = np.zeros((len(seeds),evals))
        coverages[labels[i]] = np.zeros((len(seeds),evals))
        best_perf[labels[i]] = np.zeros((len(seeds),evals))
        avg_perf[labels[i]] = np.zeros((len(seeds),evals))
        for idx,seed in enumerate(seeds):
            iterations, QD_Score, Coverage, BestReward, AvgReward = \
                get_score_from_resultsfile(resultsfolder,method,game,seed)
            qd_scores[labels[i]][idx][0:len(iterations)] = QD_Score
            times[labels[i]][idx][0:len(iterations)] = iterations
            coverages[labels[i]][idx][0:len(iterations)] = Coverage
            best_perf[labels[i]][idx][0:len(iterations)] = BestReward
            avg_perf[labels[i]][idx][0:len(iterations)] = AvgReward
    return times, qd_scores, coverages, best_perf, avg_perf

def plot(metric, resultsfolder,labels, games, scores, times, seeds, colors, markers, ext_str='', with_legend=False):
    # now get stats on the scores
    for game in games:
        score_l = {game:{}}
        score_u = {game:{}}
        score_m = {game:{}}
        #time_avg = {game:{}}

        for label in labels:
            times_list = times[game][label]
            time = np.min([len(l) for l in times_list])
            score_l[game][label] = [[] for i in range(time)]
            score_u[game][label] = [[] for i in range(time)]
            score_m[game][label] = [[] for i in range(time)]
            #time_avg[game][label] = [[] for i in times]
        for label in labels:
            times_list = times[game][label]
            time = np.min([len(l) for l in times_list])
            for idx in range(time):
                times_list = times[game][label]
                sc = [scores[game][label][s,idx] for s, seed in enumerate(seeds) if times_list[s][idx] != 0.0]
                m = np.mean(sc)
                s = np.std(sc) / np.sqrt(len(sc))
                score_l[game][label][idx] = m - s
                score_u[game][label][idx] = m + s
                score_m[game][label][idx] = m
                #time_avg[game][label][idx] = np.mean(times[game][label][:,idx])
        fig, ax = plt.subplots()

        lines=[]
        betweens=[]

        for i, label in enumerate(labels):
            steps = np.array(range(0,len(score_m[game][label])))
            line, = ax.plot(steps,score_m[game][label],marker=markers[label],color=colors[label],scaley="linear",linewidth=2)
            b = ax.fill_between(steps, score_l[game][label],  score_u[game][label],color=colors[label],alpha=0.25)
            lines.append(line)
            betweens.append(b)
        if with_legend:
            ax.legend(lines,labels,fontsize=16,framealpha=0.5)
        ax.set_xlabel("Iterations",fontsize=22)
        ax.set_ylabel(metric,fontsize=22)
        ax.set_title(game)
        fig.tight_layout()
        # fig.savefig(f"{resultsfolder}/IL_PPGA_{game}_{metric}.pdf")
        fig.savefig(f"{resultsfolder}/IL_PPGA{ext_str}_{game}_{metric}.png", dpi=600)
        fig.clf()
        # fig.show()
        plt.close()

def make_table(metric, resultsfolder,labels, games,scores,subsample=4):
    stop_eval = int(2000/subsample) # use smaller to check ongoing runs
    # table
    writefile = open(f"{resultsfolder}/table_{metric}.txt", "w")
    for label in labels:
        writefile.write(r" & " + label)
    writefile.write("\n")
    for game in games:
        writefile.write(game )
        for method, score in scores[game].items():
            score=np.array(score)
            sc = score[:, stop_eval-4:stop_eval]
            avg_score = np.mean(sc)
            final_score=avg_score# last four evaluations
            
            if metric == 'Coverage':
                final_score = format(final_score, "0.2f")
                writefile.write(r" & %s"%(final_score))
            else:
                final_score = round(final_score)
                final_score = format(final_score, ",")
                writefile.write(r" & %s"%(final_score))
        writefile.write(" \n")


if __name__ == '__main__':
    markers = {
        "GAIL": ",",
        "VAIL": ",",
        "mACGAIL_NLL_FCME": ",",
        "mCondACGAIL_NLL_FCME": ",",
        "mACGAIL_MSE_FCME": ",",
        "mCondACGAIL_MSE_FCME": ",",
        "mACGAIL_MSE_mCuriosity": ",",
        "mCondACGAIL_MSE_mCuriosity": ",",
        "mCondGAIL": ",",
        "mRegGAIL-mCuriosity": ",",
        "mRegGAIL-mEntropy": "*",
        "mRegGAIL-FCME": ",",
        "mRegGAIL-WFCME": ",",
        "mCondRegGAIL-mCuriosity": ",",
        "mCondRegGAIL-mEntropy": ",",
        "mCondRegGAIL-FCME": ",",
        "ICM": ",",
        "mCondICM": ",",
        "mRegICM-mCuriosity": ",",
        "mRegICM-mEntropy": ',',
        "mRegICM-FCME": ',',
        "mRegICM-WFCME": ',',
        "mCondRegICM-mCuriosity": ",",
        "mCondRegICM-mEntropy": ',',
        "mCondRegICM-FCME": ',',
        "GIRIL": ',',
        "PPGA-trueReward": '*'
    }
    colors = {
        "GAIL": "tab:brown",
        "VAIL": "tab:blue",
        "mACGAIL_NLL_FCME": "green",
        "mCondACGAIL_NLL_FCME": "red",
        "mACGAIL_MSE_FCME": "lightgreen",
        "mCondACGAIL_MSE_FCME": "pink",
        "mACGAIL_MSE_mCuriosity": "darkblue",
        "mCondACGAIL_MSE_mCuriosity": "orange",
        "mCondGAIL": "cyan",
        "mRegGAIL-mCuriosity": "purple",
        "mRegGAIL-mEntropy": "gold",
        "mRegGAIL-FCME": "yellow",
        "mRegGAIL-WFCME": "pink",
        "mCondRegGAIL-mCuriosity": "tab:green",
        "mCondRegGAIL-mEntropy": "tab:red",
        "mCondRegGAIL-FCME": "darkblue",
        "ICM": "gray",
        "mCondICM": "tab:purple",
        "mRegICM-mCuriosity": "tab:green",
        "mRegICM-mEntropy": "tab:red",
        "mRegICM-FCME": "darkblue",
        "mCondRegICM-mCuriosity": "lightgreen",
        "mCondRegICM-mEntropy": "yellow",
        "mCondRegICM-FCME": "orange",
        "mCondRegICM-FCME": "pink",
        "GIRIL": 'gold',
        "PPGA-trueReward": "black"
    }
    ext_str='_GAILs'
    # ext_str='_ICMs'
    if ext_str == '_GAILs':
        methods = [
                    "expert",

                    "gail",
                    # "m_acgail_AuxLoss_NLL_Bonus_fitness_cond_measure_entropy",
                    # "m_cond_acgail_AuxLoss_NLL_Bonus_fitness_cond_measure_entropy",

                    # "m_acgail_AuxLoss_MSE_Bonus_fitness_cond_measure_entropy",
                    # "m_cond_acgail_AuxLoss_MSE_Bonus_fitness_cond_measure_entropy",
                    # "m_acgail_AuxLoss_MSE_Bonus_measure_error",
                    # "m_cond_acgail_AuxLoss_MSE_Bonus_measure_error",
                    # "m_cond_gail",

                    "m_reg_gail_RegLoss_MSE_Bonus_measure_error",
                    "m_reg_gail_RegLoss_MSE_Bonus_measure_entropy",
                    # "m_reg_gail_RegLoss_MSE_Bonus_fitness_cond_measure_entropy",

                    # "m_reg_gail_measure_entropy",
                    # "m_reg_gail_fitness_cond_measure_entropy",
                    # # "m_reg_gail_weighted_fitness_cond_measure_entropy",
                    # "m_cond_reg_gail",
                    # "m_cond_reg_gail_measure_entropy",
                    # "m_cond_reg_gail_fitness_cond_measure_entropy",

                    # "vail",
                    # "giril",
                ]
        labels =  [
                    "PPGA-trueReward",

                    "GAIL",
                    # "mACGAIL_NLL_FCME",
                    # "mCondACGAIL_NLL_FCME",

                    # "mACGAIL_MSE_FCME",
                    # "mCondACGAIL_MSE_FCME",
                    # "mACGAIL_MSE_mCuriosity",
                    # "mCondACGAIL_MSE_mCuriosity",
                    # "mCondGAIL",

                    "mRegGAIL-mCuriosity",
                    "mRegGAIL-mEntropy",
                    # "mRegGAIL-FCME",

                    #
                    # "mRegGAIL-FCME",
                    # "mRegGAIL-WFCME",
                    # "mCondRegGAIL-mCuriosity",
                    # "mCondRegGAIL-mEntropy",
                    # "mCondRegGAIL-FCME",

                    # "VAIL",
                ]
    if ext_str == '_ICMs':
        methods = [
                    "expert",

                    "icm",
                    # "m_cond_icm",
                    # "m_reg_icm",
                    # "m_reg_icm_measure_entropy",
                    # "m_reg_icm_fitness_cond_measure_entropy",
                    # "m_cond_reg_icm",
                    # "m_cond_reg_icm_measure_entropy",
                    # "m_cond_reg_icm_fitness_cond_measure_entropy",
                    # "giril",
                ]
        labels =  [
                    "PPGA-trueReward",

                    "ICM",
                    # "mCondICM",
                    # "mRegICM-mCuriosity",
                    # "mRegICM-mEntropy",
                    # "mRegICM-FCME",
                    # "mCondRegICM-mCuriosity",
                    # "mCondRegICM-mEntropy",
                    # "mCondRegICM-FCME",

                    # "GIRIL",
                ]


    games = ["humanoid","halfcheetah"] #  "ant" "walker2d",
    # games = ["ant"]
    seeds=[1111] #,2222

    data_str='good_and_diverse_elite_with_measures_top500'
    # data_str='good_and_diverse_elite_with_measures_topHalfMax'
    num_demo=4
    # num_demo=8
    # num_demo=16
    # num_demo=64
    resultsfolder=f'experiments_{num_demo}_{data_str}'
    results_dict= {game: get_method_scores(resultsfolder,game,methods,labels,seeds) for game in games}
    # times, qd_scores, coverages, best_perf, avg_perf
    times_dict = {game: results_dict[game][0] for game in games}
    qd_scores_dict = {game: results_dict[game][1] for game in games}
    coverages_dict = {game: results_dict[game][2] for game in games}
    best_perf_dict = {game: results_dict[game][3] for game in games}
    avg_perf_dict = {game: results_dict[game][4] for game in games}
    make_table("QD-Score", resultsfolder,labels, games, qd_scores_dict)
    make_table("Coverage", resultsfolder,labels, games, coverages_dict)
    make_table("BestReward", resultsfolder,labels, games, best_perf_dict)
    make_table("AverageReward", resultsfolder,labels, games, avg_perf_dict)
    plot("QD-Score", resultsfolder,labels,games, qd_scores_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True)
    plot("Coverage", resultsfolder,labels,games, coverages_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True)
    plot("BestReward", resultsfolder,labels,games, best_perf_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True)
    plot("AverageReward", resultsfolder,labels,games, avg_perf_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True)

