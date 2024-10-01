import pandas as pd
import numpy as np
import csv 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.use('pdf')

def read_data(summary_file, subsample=1):
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
            try: 
                iterations, QD_Score, Coverage, BestReward, AvgReward = \
                    get_score_from_resultsfile(resultsfolder,method,game,seed)
                qd_scores[labels[i]][idx][0:len(iterations)] = QD_Score
                times[labels[i]][idx][0:len(iterations)] = iterations
                coverages[labels[i]][idx][0:len(iterations)] = Coverage
                best_perf[labels[i]][idx][0:len(iterations)] = BestReward
                avg_perf[labels[i]][idx][0:len(iterations)] = AvgReward
            except:
                print(f"Error reading {method} {game} {seed}")
    return times, qd_scores, coverages, best_perf, avg_perf

def plot(metric, resultsfolder,labels, games, scores, times, seeds, colors, markers, ext_str='', with_legend=False,format = 'png'):
    # now get stats on the scores
    print('normal plot')
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
            #legend outside
            ax.legend(lines,labels,fontsize=10,framealpha=0.5,loc='center', bbox_to_anchor=(0, -0.2))
        ax.set_xlabel("Iterations",fontsize=22)
        ax.set_ylabel(metric,fontsize=22)
        ax.set_title(game)
        fig.tight_layout()
        # fig.savefig(f"{resultsfolder}/IL_PPGA_{game}_{metric}.pdf")
        if format == 'both':
            fig.savefig(f"{resultsfolder}/figures/IL_PPGA{ext_str}_{game}_{metric}.png", dpi=600)
            fig.savefig(f"{resultsfolder}/figures/IL_PPGA{ext_str}_{game}_{metric}.pdf", dpi=600)
        else:
            fig.savefig(f"{resultsfolder}/figures/IL_PPGA{ext_str}_{game}_{metric}.{format}", dpi=600)
        fig.clf()
        fig.show()
        
        #plt.close()

def make_table(metric, resultsfolder,labels, games,scores,subsample=4):
    '''make table'''
    print('make_table')
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
        
        
def plot_combined_figure(resultsfolder, labels, games, qd_scores, coverages, best_perf, avg_perf, times, seeds, colors, markers, ext_str='', format='png'):
    print('now plotting combined figure')
    metrics = ["QD-Score", "Coverage(%)", "BestReward", "AverageReward"]
    score_dicts = [qd_scores, coverages, best_perf, avg_perf]

    # 创建大图，包含3行4列子图
    n_games = len(games)
    fig, axes = plt.subplots(n_games, 4, figsize=(20+3, 5*n_games))
    if n_games == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
     # 调整底部边距，给图例留出空间

    # 用于存储绘制的曲线以创建图例
    lines = []
    labels_legend = []

    for row, game in enumerate(games):  # 每一行是一个环境
       
        for col, metric in enumerate(metrics):  # 每一列是一个指标
            ax = axes[row, col]
            score_l = {game: {}}
            score_u = {game: {}}
            score_m = {game: {}}

            for label in labels:
                times_list = times[game][label]
                time = np.min([len(l) for l in times_list])
                score_l[game][label] = [[] for i in range(time)]
                score_u[game][label] = [[] for i in range(time)]
                score_m[game][label] = [[] for i in range(time)]

            for label in labels:
                times_list = times[game][label]
                time = np.min([len(l) for l in times_list])
                for idx in range(time):
                    sc = [score_dicts[col][game][label][s, idx] for s, seed in enumerate(seeds) if times_list[s][idx] != 0.0]
                    m = np.mean(sc)
                    s = np.std(sc) / np.sqrt(len(sc))
                    score_l[game][label][idx] = m - s
                    score_u[game][label][idx] = m + s
                    score_m[game][label][idx] = m

            for i, label in enumerate(labels):
                steps = np.array(range(0, len(score_m[game][label])))
                line, = ax.plot(steps, score_m[game][label], marker=markers[label], color=colors[label], linewidth=2)
                ax.fill_between(steps, score_l[game][label], score_u[game][label], color=colors[label], alpha=0.25)
                
                # 仅保存一次用于图例
                if row == 0 and col == 0:
                    lines.append(line)
                    labels_legend.append(label)

            # 只在每一行的第一列写 ylabel
            if col == 0:
                ax.set_ylabel(game.capitalize(), fontsize=25)  # 增大字体
            # 只在每一行的顶部写指标名称
            if row == 0:
                ax.set_title(metric, fontsize=27)  # 增大字体
            
            ax.set_xlabel("Iterations", fontsize=18)  # 增大字体
            ax.tick_params(axis='both', which='major', labelsize=14)  # 增大刻度字体

    # 创建一个总的图例
    if n_games == 1:
        fig.legend(lines, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 0.15), fontsize=25, ncol=len(labels))  # 增大图例字体
    else:    
        fig.legend(lines, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 0.1), fontsize=25, ncol=len(labels))

    # 保存大图
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    if format == 'both':
        fig.savefig(f"{resultsfolder}/combined_{ext_str}_metrics.png", dpi=600)
        fig.savefig(f"{resultsfolder}/combined_{ext_str}_metrics.pdf", dpi=600)
    else:
        fig.savefig(f"{resultsfolder}/combined_{ext_str}_metrics.{format}", dpi=600)
    
    plt.close(fig)


def make_final_metrics_csv(resultsfolder, labels, games, metrics, scores_dicts, times_dict, seeds,ext_str):
    import csv
    print('make_final_metrics_csv')
    # 打开 CSV 文件进行写入
    with open(f"{resultsfolder}/final_{ext_str}_metrics.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头：第一列是 'Game'，第二列是 'Model'，接下来是指标名称
        header = ['Game', 'Model'] + metrics
        writer.writerow(header)

        for game in games:
            for label in labels:
                row = [game, label]
                for metric_name, scores_dict in zip(metrics, scores_dicts):
                    # 获取当前游戏和模型的得分和时间
                    score = scores_dict[game][label]  # 形状：(num_seeds, num_evals)
                    times_list = times_dict[game][label]  # 每个种子的时间列表

                    final_values = []
                    for s_idx in range(len(seeds)):
                        # 获取当前种子的时间和得分
                        times_s = times_list[s_idx]
                        scores_s = score[s_idx]

                        # 找到最后一个非零的时间索引
                        non_zero_indices = np.nonzero(times_s)[0]
                        if len(non_zero_indices) == 0:
                            continue  # 该种子没有数据

                        last_idx = non_zero_indices[-1]
                        final_value = scores_s[last_idx]
                        final_values.append(final_value)

                    if len(final_values) == 0:
                        avg_final_value = np.nan  # 如果没有数据，用 NaN 表示
                    else:
                        # 计算种子间的平均值
                        avg_final_value = np.mean(final_values)

                    # 根据指标格式化最终值
                    if metric_name == 'Coverage':
                        avg_final_value = f"{avg_final_value:.2f}"
                    else:
                        avg_final_value = f"{avg_final_value:,.0f}"

                    # 将平均最终值添加到行
                    row.append(avg_final_value)
                # 写入行到 CSV 文件
                writer.writerow(row)
import seaborn as sns
import os
def line_plot(num_demos, data_str, games, methods, labels, seeds, colors, markers,_format='png'):
    """
    绘制最后一轮的数据并展示不同 seed 的 band。
    
    参数:
    - num_demos: 一个包含不同 num_demo 值的列表 (例如: [1, 2, 4])
    - data_str: 结果文件夹的基础路径的一部分，用于找到文件夹
    - games: 游戏列表
    - methods: 方法列表
    - labels: 每种方法的标签
    - seeds: 种子列表
    - colors: 每种方法的颜色
    - markers: 每种方法的标记
    """
    print('line_plot')
    metrics = ['QD-Score', 'Coverage', 'Maximum', 'Average']
    missing_files = []  # 用于记录未找到的文件
    

    for game in games:
        for metric in metrics:
            fig, ax = plt.subplots()

            for i, method in enumerate(methods):
                all_scores = []
                
                for num_demo in num_demos:
                    demo_scores = []

                    for seed in seeds:
                        # 构造 summary 文件的路径
                        summary_file = f"experiments_{num_demo}_{data_str}/IL_ppga_{game}_{method}/{seed}/summary.csv"
                        
                        if not os.path.exists(summary_file):
                            print(f"File not found: {summary_file}")
                            missing_files.append(summary_file)  # 记录文件缺失的路径
                            continue  # 文件不存在，跳过这个 seed
                        
                        try:
                            # 读取 summary.csv 文件
                            df = pd.read_csv(summary_file)
                            
                            # 获取最后一轮的数据
                            score = df[metric].iloc[-1]  # 获取最后一轮的数据
                            demo_scores.append(score)

                        except Exception as e:
                            print(f"Error reading file: {summary_file}, error: {e}")
                            missing_files.append(summary_file)  # 记录读取错误的文件
                            continue  # 出现错误时也跳过这个 seed

                    # 若存在有效数据，将其添加到 all_scores
                    if len(demo_scores) > 0:
                        all_scores.append(np.array(demo_scores))

                # 确保 all_scores 是一个二维数组，补齐缺失数据并计算均值和标准误差
                if len(all_scores) > 0:
                    max_len = max([len(demo) for demo in all_scores])  # 找出最大长度
                    padded_scores = [np.pad(demo, (0, max_len - len(demo)), constant_values=np.nan) for demo in all_scores]
                    all_scores = np.array(padded_scores)
                    
                    means = np.nanmean(all_scores, axis=1)
                    std_errors = np.nanstd(all_scores, axis=1) / np.sqrt(len(seeds))

                    # 检查 colors 和 markers 是否有足够的元素
                    if i < len(colors) and i < len(markers):
                        ax.plot(num_demos[:len(means)], means, marker=markers[labels[i]], color=colors[labels[i]], linewidth=2, label=labels[i])
                        ax.fill_between(num_demos[:len(means)], means - std_errors, means + std_errors, color=colors[labels[i]], alpha=0.25)
                    else:
                        print(f"Index {i} out of bounds for colors or markers.")
                        continue  # 跳过索引错误的情况

            # 设置标签、标题
            ax.set_xlabel("num_demos", fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_title(f'{game} - {metric}', fontsize=16)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.16), fontsize=10, ncol=len(methods))
            plt.grid(True)
            if _format == 'both':
                plt.savefig(f"{game}_{metric}_lineplot.png", dpi=300,bbox_inches="tight")
                print(f"Saved {game}_{metric}_lineplot.png")
                plt.savefig(f"{game}_{metric}_lineplot.pdf", dpi=300,bbox_inches="tight")
                print(f"Saved {game}_{metric}_lineplot.pdf")
            else:
                plt.savefig(f"{game}_{metric}_lineplot.{_format}", dpi=300,bbox_inches="tight")
                print(f"Saved {game}_{metric}_lineplot.{_format}")
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    markers = {
        "GAIL": ",",
        'GAIL': ',',
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
        "PPGA-trueReward": '*',
        "PPGA-zeroReward": '*' ,
        'gail_single_step_bonus': ',',
        'gail_archive_bonus': ',',
        'm_cond_reg_gail_single_step_bonus': ',',
        'm_cond_reg_gail_archive_bonus': ',',
        'm_reg_gail_archive_bonus': ',',
        'MCond-GAIL': ',',
        'm_cond_gail_archive_bonus': ',',
        'm_cond_gail_archive_bonus_wo_smooth': ',',
        'GIRIL': ',',
        'm_cond_gail_archive_bonus_wo_a': ',',
        'Mbo-GAIL': ',',
        'm_cond_vail_archive_bonus_wo_smooth':',',
        'VAIL':',',
        'MConbo-GAIL':',',#main
        'MConbo-VAIL':',',#main
        'MConbo-GAIL-Obs':',',
        'GAIL-Obs':',',
        'PWIL':',',
        'AIRL':',',
        'Max-Ent':','
    }
    colors = {
        "PPGA-trueReward": "black",#expert
        
        'GAIL':'lightgreen',#baseline
        'VAIL':'tab:orange',#baseline
        'GIRIL':'tab:cyan',#baseline
        'AIRL':'darkblue',#baseline
        'PWIL':'tab:pink',#baseline
        'Max-Ent':'gold',#baseline
        
        'MConbo-GAIL':'tab:olive',#main
        'MConbo-VAIL':'tab:blue',#main
        
        'MCond-GAIL':'tab:gray',#ablation
        'Mbo-GAIL':'yellow',#ablation
        


        'MConbo-GAIL-Obs':'tab:purple',#IFO
        'GAIL-Obs':'tab:red'#IFO
        }
    
    
    colors = {
    "PPGA-trueReward": "black",  # expert
    
    'GAIL': 'green',    # baseline (中等的绿色)
    'VAIL': 'darkorange',        # baseline (深橙色，具有高区分度)
    'GIRIL': 'purple',       # baseline (亮蓝色，和主模型有明显区别)
    'AIRL': 'grey',         # baseline (砖红色，区分度高)
    'PWIL': 'darkturquoise',     # baseline (深青色)
    'Max-Ent': 'goldenrod',      # baseline (深金色，替换lightgoldenrodyellow)
    
    'MConbo-GAIL': 'firebrick',  # main (鲜明的皇家蓝)
    'MConbo-VAIL': 'blue',# main (深绿色)
    
    'MCond-GAIL': 'lightblue',     # ablation (深灰色)
    'Mbo-GAIL': 'darkgoldenrod', # ablation (深金色)
    
    'MConbo-GAIL-Obs': 'royalblue',  # IFO (中等紫色)
    'GAIL-Obs': 'crimson'        # IFO (深红色)
}




    ext_str='_GAILs'
    methods_map = {
                   'all':['expert','gail','vail','giril','abgail_archive_bonus_wo_smooth','m_cond_gail','m_cond_gail_archive_bonus_wo_smooth','m_cond_vail_archive_bonus_wo_smooth'],
                   'gail_main':['expert','m_cond_gail_archive_bonus_wo_smooth','gail','giril','pwil','airl_sigmoid','irl_sigmoid'],
                   'vail_main':['expert','m_cond_vail_archive_bonus_wo_smooth','vail','giril','pwil','airl_sigmoid','irl_sigmoid'],
                   'gail_ablation_bonus':['expert',
                                    #'gail',
                                    'm_cond_gail_archive_bonus_wo_smooth',
                                    #'abgail_archive_bonus_wo_smooth',
                                    'm_cond_gail'
                                    ],
                   'gail_ablation_cond':['expert',
                                    #'gail',
                                    'm_cond_gail_archive_bonus_wo_smooth',
                                    'abgail_archive_bonus_wo_smooth'
                                    #'m_cond_gail'
                                    ],
                   'gail+vail':['expert','gail','vail','giril','pwil','airl_sigmoid','irl_sigmoid','m_cond_gail_archive_bonus_wo_smooth','m_cond_vail_archive_bonus_wo_smooth'],
                   'gail_scale':['expert','gail','m_cond_gail_archive_bonus_wo_smooth'],
                   'IFO':['expert',
                          'm_cond_gail_archive_bonus_wo_a_wo_smooth',
                          'm_cond_gail_archive_bonus_wo_smooth',
                          #'gail_wo_a',
                          #'gail'
                          ]
                   
                   
                   
                   }
    tgts = ['gail_main','vail_main','gail_ablation_bonus','gail_ablation_cond','gail_scale','IFO']
    tgts = ['IFO']
    for tgt in tgts:
        print(f"Plotting {tgt}")
        format_ = 'both'
        #format_ = 'pdf'
        

        # ext_str='_ICMs'
        if True:
            methods = [
                        "expert",
                        'gail',
                        'vail',
                        'giril',
                        'abgail_archive_bonus_wo_smooth',#ablation
                        'm_cond_gail',#ablation
                        'm_cond_gail_archive_bonus_wo_smooth',
                        'm_cond_vail_archive_bonus_wo_smooth',
                        
                        
                        
                        
                        #"zero",

                        #"gail",
                        # "m_acgail_AuxLoss_NLL_Bonus_fitness_cond_measure_entropy",
                        # "m_cond_acgail_AuxLoss_NLL_Bonus_fitness_cond_measure_entropy",

                        # "m_acgail_AuxLoss_MSE_Bonus_fitness_cond_measure_entropy",
                        # "m_cond_acgail_AuxLoss_MSE_Bonus_fitness_cond_measure_entropy",
                        # "m_acgail_AuxLoss_MSE_Bonus_measure_error",
                        # "m_cond_acgail_AuxLoss_MSE_Bonus_measure_error",
                        #"m_cond_gail",
                        
                        #"m_reg_gail_RegLoss_MSE_Bonus_measure_error",
                        #"m_reg_gail_RegLoss_MSE_Bonus_measure_entropy",
                        # "m_reg_gail_RegLoss_MSE_Bonus_fitness_cond_measure_entropy",
                        #"m_cond_reg_gail",
                        
                        #'abgail',
                        
                        #'abgail_pure'
                        #'abgail_archive_bonus',
                        
                        #'m_cond_reg_gail_RegLoss_MSE_Bonus_single_step_bonus',
                        #'m_cond_reg_gail_archive_bonus_RegLoss_MSE_Bonus_None',
                        #"m_reg_gail_archive_bonus_RegLoss_MSE_Bonus_None",
                        
                        #'m_cond_gail_archive_bonus',
                        
                        
                        #'m_cond_gail_archive_bonus_wo_a'
                        
                        # "m_reg_gail_measure_entropy",
                        # "m_reg_gail_fitness_cond_measure_entropy",
                        # # "m_reg_gail_weighted_fitness_cond_measure_entropy",
                        # "m_cond_reg_gail",
                        # "m_cond_reg_gail_measure_entropy",
                        # "m_cond_reg_gail_fitness_cond_measure_entropy",

                        # "vail",
                        # "giril",
                    ]
            methods = methods_map[tgt]
            if tgt == 'vail_main':
                ext_str='_VAILs'
            if tgt == 'gail_main':
                ext_str='_GAILs'
            if tgt == 'gail_ablation_bonus':
                ext_str='_GAILs_ablation_bonus'
            if tgt == 'gail_ablation_cond':
                ext_str='_GAILs_ablation_cond'
            if tgt == 'all':
                ext_str='_all'
            if tgt == 'gail+vail':
                ext_str='_GAILs+VAILs'
            if tgt == 'gail_scale':
                ext_str='_GAILs_scalability'
            if tgt == 'IFO':
                ext_str='_IFO'
                
            
            
            mapp = {'expert':'PPGA-trueReward',
                    'gail':'GAIL',
                    'giril':'GIRIL',
                    'abgail':'gail_archive_bonus',
                    'abgail_archive_bonus':'gail_archive_bonus',
                    'abgail_archive_bonus_wo_smooth':'Mbo-GAIL',
                    'abgail_pure':'gail_baseline',
                    'm_cond_reg_gail_single_step_bonus':'m_cond_reg_gail_single_step_bonus',
                    'm_cond_reg_gail_archive_bonus':'m_cond_reg_gail_archive_bonus',
                    'm_reg_gail_archive_bonus':'m_reg_gail_archive_bonus',
                    'm_cond_gail':'MCond-GAIL',
                    'm_cond_gail_archive_bonus':'m_cond_gail_archive_bonus',
                    'm_cond_gail_archive_bonus_wo_smooth':'MConbo-GAIL',#main
                    'm_cond_gail_archive_bonus_wo_a':'m_cond_gail_archive_bonus_wo_a',
                    'm_cond_vail_archive_bonus_wo_smooth':'MConbo-VAIL',#main
                    'vail':'VAIL',
                    'm_cond_gail_archive_bonus_wo_a_wo_smooth':'MConbo-GAIL-Obs',
                    'gail_wo_a':'GAIL-Obs',
                    'pwil':'PWIL',
                    'airl_sigmoid':'AIRL',
                    'irl_sigmoid':'Max-Ent'
                    
                    
                    }
            labels =  [mapp[method] for method in methods
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


        # games = ["humanoid","halfcheetah"] #  "ant" "walker2d",
        
        games = ["halfcheetah","walker2d","humanoid"]
        num_demos=[4]
        if tgt == 'gail_ablation_bonus':
            games = ["walker2d"]
        if tgt == 'gail_ablation_cond':
            games = ["halfcheetah"]
        if tgt == 'gail_scale':
            games = ["walker2d"]
            num_demos = [1,2,4]
        if tgt == 'IFO':
            pass
        seeds=[1111,2222,3333] #,2222
        
        #num_demos = [1,2,4]
        data_str='good_and_diverse_elite_with_measures_top500'
        # data_str='good_and_diverse_elite_with_measures_topHalfMax'
        # num_demo=4
        
        # num_demo=16
        # num_demo=64
        for num_demo in num_demos:
            resultsfolder=f'experiments_{num_demo}_{data_str}'
            if 'ant' in games:
                resultsfolder = 'experiments_4x50_good_and_diverse_elite_with_measures_top500'#for ant
            
            
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
            
            if ext_str == '_GAILs+VAILs' or ext_str == '_GAILs_scalability' or ext_str == '_IFO':
                metrics = ['QD-Score', 'Coverage', 'BestReward', 'AverageReward']
                scores_dicts = [qd_scores_dict, coverages_dict, best_perf_dict, avg_perf_dict]
                make_final_metrics_csv(resultsfolder, labels, games, metrics, scores_dicts, times_dict, seeds,ext_str)
            
            plot_combined_figure(resultsfolder, labels, games, qd_scores_dict, coverages_dict, best_perf_dict, avg_perf_dict, times_dict, seeds, colors, markers, ext_str, format=format_)
            
            plot("QD-Score", resultsfolder,labels,games, qd_scores_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True,format=format_)
            """plot("Coverage", resultsfolder,labels,games, coverages_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True,format=format_)
            plot("BestReward", resultsfolder,labels,games, best_perf_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True,format=format_)
            plot("AverageReward", resultsfolder,labels,games, avg_perf_dict, times_dict, seeds, colors, markers, ext_str, with_legend=True,format=format_) """
        
        if ext_str == '_GAILs_scalability':
            line_plot(num_demos, data_str, games, methods, labels, seeds, colors, markers,_format=format_)

    
    
    
    
    
    
    



# Usage in the main function when ext_str == '_all':

