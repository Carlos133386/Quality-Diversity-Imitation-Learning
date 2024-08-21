import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle



def visualize(num_demo=4, env_name='walker2d', topk=150):
    demo_dir = 'trajs_good_and_diverse_elite_with_measures' + f'_top{topk}'
    file_name=f'{demo_dir}/{num_demo}episodes/trajs_ppga_{env_name}.pt'
    traj_data = pickle.load(open(file_name, 'rb'))
    demonstrator_measures = traj_data['demonstrator_measures']
    full_occupied_measures = traj_data['full_occupied_measures']
    topk_occupied_measures = traj_data['topk_occupied_measures']
    traj_lengths = traj_data['lengths']

    traj_returns = traj_data['returns']
    demostrator_returns = traj_data['demonstrator_returns']

    # print(f'{env_name}, {num_demo}demos, top{topk} | \
    #       traj lengths, min({traj_lengths.min()}), mean+std({traj_lengths.mean()}+{traj_lengths.std()}), max({traj_lengths.max()}) | \
    #       traj returns, min({traj_returns.min()}), mean+std({traj_returns.mean()}+{traj_returns.std()}), max({traj_returns.max()}) | \
    #       demostrator returns, min({demostrator_returns.min()}), mean+std({demostrator_returns.mean()}+{demostrator_returns.std()}), max({demostrator_returns.max()})')
    
    print(f'{env_name}, {num_demo}demos, top{topk}') 
    print(f'traj lengths, {traj_lengths.min():.1f} & {traj_lengths.max():.1f} & {traj_lengths.mean():.1f} & {traj_lengths.std():.1f}  \\')
    print(f'traj returns, {traj_returns.min():.1f} & {traj_returns.max():.1f} & {traj_returns.mean():.1f} & {traj_returns.std():.1f} \\')
    print(f'demostrator returns, {demostrator_returns.min():.1f} & {demostrator_returns.max():.1f} & {demostrator_returns.mean():.1f} & {demostrator_returns.std():.1f} \\')
    fig, ax = plt.subplots()
    for measure in full_occupied_measures :
        x,y=measure[0], measure[1]
        ax.plot(x, y, 'go')
    for measure in topk_occupied_measures :
        x,y=measure[0], measure[1]
        ax.plot(x, y, 'bo')
    for measure in demonstrator_measures:
        x,y=measure[0], measure[1]
        ax.plot(x, y, 'ro')

    labels = ['full_occupied_measures', 'topk_occupied_measures', 'demonstrator_measures']
    
    handles = [
        Line2D([0], [0], c='g', marker='o', linestyle=''),
        Line2D([0], [0], c='b', marker='o', linestyle=''),
        Line2D([0], [0], c='r', marker='o', linestyle='')]
    ax.legend(handles,labels, loc="best")
    ax.set_title(f'{env_name}, {num_demo} demos (red), top{topk} (blue)')
    plt.savefig(file_name.replace('pt', 'png'))
    plt.close()
    if env_name == 'ant':
        fig, ax = plt.subplots()
        for measure in full_occupied_measures :
            x,y=measure[2], measure[3]
            ax.plot(x, y, 'go')
        for measure in topk_occupied_measures :
            x,y=measure[2], measure[3]
            ax.plot(x, y, 'bo')
        for measure in demonstrator_measures:
            x,y=measure[2], measure[3]
            ax.plot(x, y, 'ro')

        labels = ['full_occupied_measures', 'topk_occupied_measures', 'demonstrator_measures']
        
        handles = [
            Line2D([0], [0], c='g', marker='o', linestyle=''),
            Line2D([0], [0], c='b', marker='o', linestyle=''),
            Line2D([0], [0], c='r', marker='o', linestyle='')]
        ax.legend(handles,labels, loc="best")
        ax.set_title(f'{env_name}, {num_demo} demos (red), top{topk} (blue)')
        plt.savefig(file_name.replace('.pt', '_last2dims.png'))

    plt.close()

if __name__ == '__main__':
    # topk=100
    # topk=150
    # topk=200
    # topk=250
    # topk=300
    # topk=350
    # topk=400
    # topk=450
    # topk=500
    # topk=600
    # topk=700
    # topk=800
    # topk=900
    # topk=1000
    for topk in [500]: #'HalfMax'
        for env_name in ['ant']: # , 
            for num_demo in [8]: #, 4, 8, 16, 32, 64
                visualize(num_demo, env_name, topk)