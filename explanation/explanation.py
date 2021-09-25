
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import shap
from tqdm import tqdm

import matplotlib
from pathlib import Path
import pickle
import pandas  as pd
from shutil import copyfile
import os.path

import argparse
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--model_path',type=str,  default="../model/dqn/dqn_agent_intersection_1_1.pickle", help='path of model file')
parser.add_argument('--states_path',type=str,  default="../scripts/buffer.csv", help='path of states file')
parser.add_argument('--output_path', type=str, default="dqn/", help='path to save all informations about the model and shap explanation')
parser.add_argument('--explainer_type', type=str,  default="deep", help='define explainer type for shap')
parser.add_argument('--force_new', type=bool, default=False, help='calculate new shap values even if we already have them')
parser.add_argument('--mult_out_num', type=int, default=1, help='num of multplots we plot - 0 means none')
parser.add_argument('--summary_plot', type=bool, default=True, help='if we plot summary')
parser.add_argument('--forceplot_num',type=int, default=1, help='num of force plots for best action in x states - 0 means none')
parser.add_argument('--num_states', type=int, default=5000, help='num of states we use to calculate shap values')
parser.add_argument('--isx', type=int, default=16, help='image size x')
parser.add_argument('--isy', type=int, default=8, help='image size y')
args = parser.parse_args()

def convert_list_string_to_list_float(strings):
    # Converting string to list
    values = []
    for s in strings:
        value = []
        for x in s.split( ):
            try:
                value.append(float(x.strip(' []')))
            except:
                pass
        values.append(np.array(value).reshape(8))
    return values

def load_states(path,num_states=1000):
    buffer = pd.read_csv(f"{path}",sep=";")
    states = convert_list_string_to_list_float(np.random.choice(buffer['actual_state'],num_states))
    return np.array(states)

def load_model_to_be_explained(path):
    return pickle.load(open(f"{path}", "rb"))

class shap_explainer_model:

    def __init__ (self, path:str="out/", auto_save:bool=True,force_new:bool=False,explainer = None):
        #Define Vars
        self.model_explained = None
        self.shap_values = None
        self.expected_value = None
        self._explainer = None
        self._explainer_func = None
        self.explainer_type = None
        self.possibilits = None

        self.path = path
        self.auto_save = auto_save
        self.force_new = force_new

        self.load_error = False #if load_error, is  to create new_shap_values

        if explainer:
            self.set_explainer(explainer)
        
        #Create Path to save 
        Path(self.path).mkdir(parents=True, exist_ok=True)

        #Create or Load shap_explainer_model
        if not force_new:
            try:
                self.update(self.load_model(self.path))
            except:
                self.load_error = True
                pass
        
        self.save_model(self.path)
        #end
    
    def get_shap_values(self,model_to_be_explained,possibilits):
        if self._explainer_func == None:
            raise Exception("Explainer iqual None... Before get shap values, run set_explainer(type)")
        
        if self.load_error or self.force_new or self.shap_values == None or self.expected_value is None:
            print("Creating shap values")
            self.model_explained = model_to_be_explained
            self.possibilits = possibilits
            self._explainer = self._explainer_func(self.model_explained,possibilits)
            self.shap_values = self._explainer.shap_values(possibilits)
            self.expected_value = self._explainer.expected_value

        self._explainer = None
        self.save_model(self.path) if self.auto_save else None

        return self.shap_values

    def _get_real_explainer_type(self,type):
        type_lower = type.lower()
        if "deep" in type_lower:
            return "DeepExplainer"
        elif "tree" in type_lower:
            return "TreeExplainer"
        elif "linear" in type_lower:
            return "LinearExplainer"
        elif "kernel" in type_lower:   
            return "KernelExplainer"
        else:
            return None

    def set_explainer(self,type):
        self.explainer_type = self._get_real_explainer_type(type)
        self._explainer_func = getattr(shap, self.explainer_type)

        return self._explainer

    def save_model(self,path=""):
        pickle.dump(self, file = open(f"{path}shap_model.pickle", "wb"))

    def load_model(self,path=""):
        return pickle.load(open(f"{path}shap_model.pickle", "rb"))
    
    def verify_load_dif(self,new_class):
        if self.model_explained != None and self.model_explained != new_class.model_explained:
            self.load_error = True
            return "dif in model_explained"

        # if self.shap_values != None and self.shap_values != new_class.shap_values:
        #     self.load_error = True
        #     return "dif in shap_values"

        # if self.expected_value != None and self.expected_value != new_class.expected_value:
        #     self.load_error = True
        #     return "dif in expected_value"

        if self._explainer_func != None and self._explainer_func != new_class._explainer_func:
            self.load_error = True
            return "dif in _explainer_func"

        if self.explainer_type != None and self.explainer_type != new_class.explainer_type:
            self.load_error = True
            return "dif in explainer_type"

        if self.possibilits != None and self.possibilits != new_class.possibilits:
            self.load_error = True
            return "dif in possibilits"
        
        return "no dif"

    def update(self,new_class):

        print(self.verify_load_dif(new_class))
        self.model_explained = new_class.model_explained
        self.shap_values = new_class.shap_values
        self.expected_value = new_class.expected_value
        self._explainer = new_class._explainer
        self._explainer_func = new_class._explainer_func
        self.explainer_type = new_class.explainer_type
        self.possibilits = new_class.possibilits

        self.path = new_class.path 
        self.auto_save = new_class.auto_save
        self.force_new = new_class.force_new

    def model_predict(self,*args):
        return self.model_explained.predict(*args)


#Cria agente shap (ou faz load do modelo ja criado)
shap_model = shap_explainer_model(path=args.output_path,force_new = args.force_new,explainer = "deep")
##or dont use explainer as parameter and use:
#shap_model.set_explainer("deep")

#Gera os estados
samples = load_states(args.states_path,num_states=args.num_states)

#Carrega modelo do agente
model = load_model_to_be_explained(args.model_path)

#Gera valores shap (ou pega os que ja estão carregados)
shap_model.get_shap_values(model,samples)

#use only vars from shap_model class.
del samples
del model
samples = shap_model.possibilits
predictions = shap_model.model_predict(samples)
shap_values = shap_model.shap_values
expected_value = shap_model.expected_value

######
#### Começa plots
######

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from colour import Color
sns.set_theme(style="darkgrid")


def plot_multi_output(runs = 1):
    #Create Path to save 
    Path(f"{args.output_path}multioutput").mkdir(parents=True, exist_ok=True)

    def hex_to_rgb(hex_color):
        hex_color = hex_color.split('#')[1]
        rgb = list(int(hex_color[i:i+2], 16)/256 for i in (0, 2, 4))
        rgb.append(1)
        return rgb
    def create_tab_b(total=8,num = 1):
        #cores_hex = ['#2cbdfe','#3aacf6','#489bee','#568ae6','#6379de', '#6e6cd8',
        #            '#7168d7', '#7f57cf','#8d46c7','#9739c1','#9b35bf','#a924b7', '#b317b1']
        
        #cores_hex = ['#322e2f','#12a4d9','#12a4d9','#b20238','#d9138a','#e2d810','#fbcbc9','#6b7b8c']
        cores_hex = ['#322e2f','#375f9f','#12a4d9','#3caea3','#b9d604','#f6d55c','#ed553b','#b20238']
        colors = []
        sub_total = 0
        jump_total = 0
        while sub_total < total:
            for sub_num in range(num):
                if sub_total >= total:
                    break
                colors.append(hex_to_rgb(cores_hex[jump_total]))
                sub_total += 1 
            jump_total += 1

        return colors

    def add_subplot_axes(ax,rect,axisbg='w'):
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        
        subax = fig.add_axes([x,y,width,height])
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
        #subax.axis('off')
        
        plt.title("Actions")
        
        hexbins = ax.hexbin(list(range(0,8)),[-1]*8, C=colors, 
                        bins=8, gridsize=8,cmap=cm)
        cmin, cmax = hexbins.get_clim()
        below = 0.25 * (cmax - cmin) + cmin
        above = 0.75 * (cmax - cmin) + cmin

        cbar = fig.colorbar(hexbins, cax=subax, orientation='vertical')
        #subax.xaxis.set_ticks_position('top')

        return subax


    #red = Color("lightblue")
    #colors = list(red.range_to(Color("darkblue"),60))
    #colors = [color.rgb for color in colors]
    colors = create_tab_b()

    cm = LinearSegmentedColormap.from_list(
            'cmap_name', colors, N=8)

    for row_index in range(runs):

        state = [np.ceil(s/.025) for s in shap_model.possibilits[row_index]]

        shap.multioutput_decision_plot(
            base_values = list(shap_model.expected_value), 
            shap_values = list(shap_model.shap_values),
            row_index = row_index,
            features=[f'F0: {state[0]}',f'F1: {state[1]}',f'F2: {state[2]}',f'F3: {state[3]}',f'F4: {state[4]}',f'F5: {state[5]}',f'F6: {state[6]}',f'F7: {state[7]}'], #None,
            feature_names=None,
            feature_order="importance", #None, #
            feature_display_range=None,
            highlight= None,#[np.argmax(predictions[row_index])],# None,#
            link="identity",
            plot_color=plt.get_cmap("tab20c"),#None,
            axis_color="#333333",
            y_demarc_color="#333333",
            alpha=None,
            color_bar=False,#True,
            auto_size_plot=True,
            #title=f'F0:{possibilits[row_index][0]} | F1:{possibilits[row_index][1]} | F2:{possibilits[row_index][2]} | Action={actions[row_index]}',#None,
            xlim=None,
            show=False,#True,
            return_objects=False,
            ignore_warnings=False,
            new_base_value=None,
            legend_labels=None,
            legend_location="best",
        )

        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(args.isx,args.isy)
        #fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(False)
        #plt.title(f'F0:{possibilits[row_index][0]} | F1:{possibilits[row_index][1]} | F2:{possibilits[row_index][2]} | Action={actions[row_index]}')

        for idx,a in enumerate(plt.gca().get_lines()[8:]):
            a.set_color(colors[idx])
            a.set_linewidth(1)
            a.set_alpha(1)

            #Essa marca alguma linha de fase desejada
            #a.set_alpha(.5)
            if idx in [np.argmax(predictions[row_index])]:
                #a.set_alpha(1)
                a.set_linewidth(1.5)
                a.set_linestyle("-.")


        rect = [0.93,0.05,.05,0.7]
        ax1 = add_subplot_axes(ax,rect)

        plt.savefig(f"{args.output_path}multioutput/multioutput_{row_index}.pdf",bbox_inches = 'tight')
        plt.close()

def plot_summary_per_action():
    
    #Create Path to save 
    Path(f"{args.output_path}summary_action").mkdir(parents=True, exist_ok=True)

    for action in range(8):
        shap.summary_plot(
            np.array(shap_model.shap_values)[action,:,:],
            np.array(shap_model.possibilits),
            show=False#True,
        )
        fig = plt.gcf()
        fig.set_size_inches(args.isx,args.isy)
        plt.savefig(f"{args.output_path}summary_action/summary_{action}.pdf",bbox_inches = 'tight')
        plt.close()

def plot_force_plot_best_action(runs = 1):#Create Path to save 
    Path(f"{args.output_path}forceplot").mkdir(parents=True, exist_ok=True)

    for row_index in range(runs):
        state = [np.ceil(s/.025) for s in shap_model.possibilits[row_index]]
        best_action = np.argmax(predictions[row_index])

        shap.force_plot(
            shap_model.expected_value[best_action], 
            shap_model.shap_values[best_action][row_index],
            state, 
            matplotlib=True,
            show=False,
            #feature_names=['Actual','Next','Others'],
            feature_names=[f'F0: {state[0]}',f'F1: {state[1]}',f'F2: {state[2]}',f'F3: {state[3]}',f'F4: {state[4]}',f'F5: {state[5]}',f'F6: {state[6]}',f'F7: {state[7]}'], #None,
            
            )
        ##ax = plt.gca().get_xlim()
        ##ax_s = ax[1] - ax[0]
        ##plt.xlim([ax[0]-.2*ax_s, ax[1]+.2*ax_s])
        ##plt.ylim([-.3, .3])

        fig = plt.gcf()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(False)
        plt.subplots_adjust(top=0.5)
        fig.set_size_inches(args.isx,args.isy)
        plt.savefig(f"{args.output_path}forceplot/forceplot_{row_index}_{best_action}.pdf",bbox_inches = 'tight')
        plt.close()
    pass

#plot_multi_output(args.mult_out_num)
#plot_summary_per_action() if args.summary_plot else None

plot_force_plot_best_action(args.forceplot_num) 