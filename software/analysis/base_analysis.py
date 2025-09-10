import os
import re
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def extract_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        return None
    
    data = []
    for line in lines[-4:]:
        try:
            # Find dictionary-like structure
            match = re.search(r'\{.*\}', line)
            if match:
                d = eval(match.group())
                if all(key in d for key in ['E', 'A', 'C', 'N', 'O']):
                    data.append(d)
                else:
                    return None
            else:
                return None
        except:
            return None
    
    return data if len(data) == 4 else None

def calculate_correlation(x, y):
    r, p = stats.pearsonr(x, y) #Pearson correlation = how strong is the LINEARITY
    slope, _, _, _, _ = stats.linregress(x, y)
    n = len(x)
    r_z = np.arctanh(r) #Applying Fisher's z-transformation bc the sampling distribution of r isn't normal
    se = 1/np.sqrt(n-3)
    z = stats.norm.ppf((1+0.95)/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z)) #Transform bounds back to r-space using inverse of Fisher's transformation
    return r, (lo, hi), slope

# Updated percentiles to be 0-100
percentiles = {"E": [0.0, 0.7571588198996014, 1.4592818665465885, 2.357991027501209, 3.4136772235286275, 4.912359700471973, 6.458573072497123, 8.332707926819099, 10.365904504594653, 12.717016060439285, 15.23448574907023, 18.10802021313854, 21.002610029852736, 24.342489284701724, 27.62983439235503, 31.20465802771802, 34.76676506395824, 38.585288771034506, 42.385675689197974, 46.435599806540914, 50.300195126832435, 54.600073381031, 58.39462317172829, 62.31196110805357, 65.90471306349127, 69.56834442387552, 72.94470572538818, 76.2651973782959, 79.26318773869681, 82.2436667166992, 84.87099948299728, 87.39597404979904, 89.58447991194276, 91.61225629992829, 93.31315354980737, 94.88125614983073, 96.14603659045045, 97.31971614883007, 98.17610613565483, 98.9013692233285, 99.42545988225679], 
                "A": [0.0, 0.06587615283267458, 0.11674254932372709, 0.20221477293574158, 0.3001951268324411, 0.4786444522272811, 0.6606377478694485, 0.8989176297926986, 1.160337552742616, 1.5259918947315754, 1.9106169009856406, 2.4211570854388684, 2.990277013392038, 3.7309667950834706, 4.550249328730342, 5.547981187772052, 6.678507696669501, 8.101515985390503, 9.63522122713097, 11.513316989376428, 13.606761061356549, 16.24952052167242, 19.079901936258565, 22.404562965927855, 26.008363769783692, 30.221310518503696, 34.64898017044412, 39.57092943746769, 44.85165357482364, 50.56891145911509, 56.17401310852054, 62.085355481062685, 67.60415103150382, 73.08083587665315, 77.89333900368572, 82.54240256166509, 86.54166875135505, 90.54343656710195, 93.50369406781074, 96.21837527726356, 98.15400843881856], 
                "C": [0.0, 0.026267073597838593, 0.052534147195677186, 0.10131585530594887, 0.18136788912793317, 0.3554393689231334, 0.5712046163339505, 0.9087156651823686, 1.3435816614132519, 1.980870899417955, 2.7793065492570173, 3.880230483147379, 5.215681859875586, 6.944722402895215, 8.97896132486116, 11.483505945531263, 14.308884108003536, 17.681909908106935, 21.305306782742115, 25.43236437017395, 29.816047097279895, 34.75258918296893, 39.64326812428078, 44.826845783092345, 49.88659295208552, 55.21026167008556, 60.11866046263405, 65.05770417438002, 69.58981671419757, 74.13547972849018, 78.13787295075132, 81.93450742982938, 85.13325328129952, 88.20504161038008, 90.66830940111072, 92.93811811010491, 94.7292823668718, 96.44018612097864, 97.59364420206468, 98.6178516035423, 99.25555777922315], 
                "N": [0.0, 0.45904838144794113, 0.9080902586681342, 1.5412101199112758, 2.3192158236186855, 3.5842047330765, 4.917362952585847, 6.6291005820449955, 8.520329881089376, 10.982346858791548, 13.478761194776606, 16.416087123296812, 19.472240289521523, 22.96409332732943, 26.558929971147915, 30.489192975434033, 34.39569053217925, 38.6205200046697, 42.744450559530364, 47.09769683627692, 51.269783692733604, 55.75019596070779, 59.84952719267523, 63.97908640616401, 67.82929737662813, 71.73558646453529, 75.19262520638415, 78.57357282233451, 81.65411684261437, 84.59165123997265, 87.07743366521572, 89.39790030186288, 91.36084288120611, 93.24560964627008, 94.66674171544838, 95.95820616734211, 96.97803572322009, 98.00578709494505, 98.65224896182518, 99.19843731758976, 99.5709711312353], 
                "O": [0.0, 0.006879471656576775, 0.009798035389669953, 0.01688597588432481, 0.025850135921682428, 0.04648855089141275, 0.06754390353729924, 0.10944613999099415, 0.1507229699304548, 0.22827337769550207, 0.32041660412601525, 0.45425359817214517, 0.6310351728623605, 0.9566634979403278, 1.2914644518937308, 1.7928320074715232, 2.4215740231150247, 3.359475325628325, 4.478744517269559, 6.045387835426361, 7.9170210636914, 10.512458097763547, 13.37577758876603, 16.875552442420908, 20.6805256750221, 25.168025883490934, 29.918613765614317, 35.21497306582612, 40.74064808792382, 46.86963192741949, 52.7936908990844, 59.05672020146429, 64.90114407698337, 70.78163306148997, 76.04588816063774, 81.07749203649038, 85.49056886976535, 89.61575023765448, 92.88996180850887, 95.71242140724804, 97.84714230916762]}

def calculate_percentile(trait, score):
    return percentiles[trait][score]

experiment_name = "llama8b_3"

data = []
for j in range(1, 1001):
    file_path = f"output_{experiment_name}/output_{j}.txt"
    if os.path.exists(file_path):
        result = extract_data(file_path)
        if result:
            data.append(result)

df = pd.DataFrame(data, columns=['bot_initial', 'bot_shift', 'user_initial', 'user_shift'])

traits = ['E', 'A', 'C', 'N', 'O']
correlations = {}
for bot_trait in traits:
    for user_trait in traits:
        x = df['user_initial'].apply(lambda d: d[user_trait])
        y = df['bot_shift'].apply(lambda d: d[bot_trait])
        r, ci, slope = calculate_correlation(x, y)
        correlations[f"{user_trait}-{bot_trait}"] = (r, ci, slope)

def print_pair_correlations():
    for pair, (r, ci, slope) in correlations.items():
        print(f"Correlation between User {pair[0]} and Bot {pair[2]} shift:")
        print(f"  r = {r:.4f}")
        print(f"  95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
        print(f"  Slope: {slope:.4f}")
        print()

def create_correlation_heatmap():
    corr_matrix = np.zeros((5, 5))
    for i, bot_trait in enumerate(traits):
        for j, user_trait in enumerate(traits):
            corr_matrix[i, j] = correlations[f"{user_trait}-{bot_trait}"][0]

    plt.figure(figsize=(12, 10))

    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=traits, yticklabels=traits, annot=False)

    # Add annotations manually
    for i in range(5):
        for j in range(5):
            r = corr_matrix[i, j]
            ci = correlations[f"{traits[j]}-{traits[i]}"][1]
            
            text = f"{r:.3f}\n({ci[0]:.3f}, {ci[1]:.3f})"
            
            plt.text(j + 0.5, i + 0.5, text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black' if abs(r) < 0.5 else 'white')

    plt.title(f"{experiment_name} Correlation Heatmap: User Initial vs Bot Shift\nwith 95% Confidence Intervals")
    plt.xlabel("User Initial Trait")
    plt.ylabel("Bot Shift Trait")

    plt.tight_layout()
    plt.savefig(f'{experiment_name}_correlation_ci_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved as '{experiment_name}_correlation_ci_heatmap.png'")

def create_scatterplots():
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle(f"{experiment_name} Raw Data Scatter Plots: User Initial vs Bot Shift", y=1.02, fontsize=16)

    for i, bot_trait in enumerate(traits):
        for j, user_trait in enumerate(traits):
            ax = axes[i, j]
            
            x = df['user_initial'].apply(lambda d: d[user_trait])
            y = df['bot_shift'].apply(lambda d: d[bot_trait])

            x_percentiles = x.apply(lambda score: calculate_percentile(user_trait, score))

            ax.scatter(x_percentiles, y, alpha=0.1, s=20)
            ax.set_xlim(0, 100)

            r = correlations[f"{user_trait}-{bot_trait}"][0]
            ax.text(0.05, 0.95, f'r = {r:.3f}', 
                    transform=ax.transAxes, 
                    verticalalignment='top')
            
            if i == 4:
                ax.set_xlabel(f'User Initial {user_trait}')
            if j == 0:
                ax.set_ylabel(f'Bot Shift {bot_trait}')
                
            ax.grid(True, alpha=0.3)
            
            ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(f'{experiment_name}_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scatter plots saved as '{experiment_name}_scatter_plots.png'")

def print_avg_shifts_and_std():
    for trait in traits:
        shifts = df['bot_shift'].apply(lambda d: d[trait])
        avg_shift = np.mean(shifts)
        std_shift = np.std(shifts)
        print(f"Bot {trait} shift:")
        print(f"  Average: {avg_shift:.4f}")
        print(f"  Standard Deviation: {std_shift:.4f}")
        print()

        initials = df['user_initial'].apply(lambda d: d[trait])
        avg_initial = np.mean(initials)
        std_initial = np.std(initials)
        print(f"User {trait} initial:")
        print(f"  Average: {avg_initial:.4f}")
        print(f"  Standard Deviation: {std_initial:.4f}")
        print()


print_pair_correlations()
create_correlation_heatmap()
create_scatterplots()
print_avg_shifts_and_std()